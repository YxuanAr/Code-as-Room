"""Stage1 runner script - spatial semantic analysis"""
import os
import sys
import json
import base64
import argparse

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from memory import Memory
from prompt_manager import PromptManager
from validators import validate_stage1_schema, extract_json_from_response
from base import ValidationResult

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class Stage1Runner:
    """Stage1 runner - spatial semantic analysis"""

    def __init__(
        self,
        image_path: str,
        output_dir: str = "./output",
        max_iterations: int = 3,
        verbose: bool = True,
        model: str = "[L]gemini-3.1-pro-preview",
        base_url: str = None,
        api_key: str = None,
        memory_file: str = "agent_memory.jsonl"
    ):
        self.image_path = image_path
        self.output_dir = output_dir
        self.max_iterations = max_iterations
        self.verbose = verbose

        # LLM
        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url or os.environ.get("SCENEGEN_BASE_URL"),
            api_key=api_key or os.environ.get("SCENEGEN_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            temperature=0.7,
            timeout=600,
            request_timeout=600,
            max_retries=3,
        )

        # Memory & Prompt
        self.memory = Memory(workspace_dir=parent_dir, memory_file=memory_file)
        self.prompts = PromptManager()
        self.task_prompt = self.prompts.get("Stage1_task")
        self.fix_template = self.prompts.get("Stage1_fix_template")

        # Scene-aware prompt routing: append lab-specific addendum if the
        # scene_classifier (run earlier in unified_pipeline) tagged this image
        # as a lab. The base prompt is preserved; the addendum is appended
        # so the full Stage1 base contract still applies.
        self.scene_type_info = self._load_scene_type_info()
        self.task_prompt = self._apply_scene_type_addendum(
            base_prompt=self.task_prompt,
            scene_info=self.scene_type_info,
        )

        # State
        self.current_output = None
        self.parsed_json = None
        self.iteration = 0

    def _load_scene_type_info(self) -> dict:
        """Read scene_type record from Memory (written by scene_classifier).

        Returns a fallback dict when no record exists or the helper module
        is unavailable; never raises so Stage1 still runs in legacy setups.
        """
        try:
            from scene_classifier import read_scene_type  # type: ignore
            return read_scene_type(self.memory)
        except Exception as exc:
            if self.verbose:
                print(f"Stage1: cannot read scene_type ({exc}); using base prompt")
            return {
                "scene_type": "other",
                "confidence": 0.0,
                "reasoning": "scene_classifier unavailable",
                "lab_subtype": None,
                "source": "fallback",
            }

    def _apply_scene_type_addendum(self, base_prompt: str, scene_info: dict) -> str:
        """Append scene-specific addendum to base_prompt when applicable.

        Routing:
          - lab (confidence >= 0.5)            -> Stage1_task_lab_addendum
          - industrial (confidence >= 0.5)     -> no addendum (use base prompt)
          - residential / office / retail / other / low-confidence lab
                                              -> Stage1_task_residential_addendum
        """
        scene_type = (scene_info or {}).get("scene_type", "other")
        confidence = float((scene_info or {}).get("confidence", 0.0) or 0.0)

        addendum_name = None
        route_label = None

        if scene_type == "lab" and confidence >= 0.5:
            addendum_name = "Stage1_task_lab_addendum"
            subtype = scene_info.get("lab_subtype") or "general"
            route_label = f"lab (subtype={subtype}, confidence={confidence:.2f})"
        elif scene_type == "industrial" and confidence >= 0.5:
            route_label = f"industrial (confidence={confidence:.2f}) - base prompt only"
        else:
            addendum_name = "Stage1_task_residential_addendum"
            route_label = (
                f"residential/office (scene_type={scene_type}, "
                f"confidence={confidence:.2f})"
            )

        if addendum_name is None:
            if self.verbose:
                print(f"Stage1: route -> {route_label}")
            return base_prompt

        try:
            addendum = self.prompts.get(addendum_name)
        except Exception as exc:
            if self.verbose:
                print(f"Stage1: failed to load {addendum_name} ({exc}); using base prompt")
            return base_prompt

        if self.verbose:
            print(f"Stage1: route -> {route_label} -> {addendum_name}")
        return base_prompt.rstrip() + "\n\n" + addendum.lstrip()

    def _log(self, msg: str, level: str = "info"):
        if self.verbose:
            prefix = {"info": "[i]", "success": "[OK]", "warning": "[!]", "error": "[X]", "step": "[>]"}.get(level, "")
            print(f"{prefix} {msg}")

    def _encode_image(self, path: str) -> tuple:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}.get(ext, "image/png")
        return b64, mime

    def _generate(self, fix_instructions: str = None) -> str:
        """Call the LLM to generate output"""
        self._log(f"Generating JSON (iteration {self.iteration}/{self.max_iterations})", "step")

        system_content = self.task_prompt
        if fix_instructions:
            system_content += f"\n\n[Fix requirements]\n{fix_instructions}"

        b64, mime = self._encode_image(self.image_path)

        user_text = "Please analyze the image and output the required JSON."
        if fix_instructions:
            user_text = f"Please fix the following issues and re-output:\n{fix_instructions}"

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": user_text}
            ])
        ]

        import time as _time
        max_api_retries = 5
        for attempt in range(1, max_api_retries + 1):
            try:
                response = self.llm.invoke(messages)
                json_str = extract_json_from_response(response.content)
                self.current_output = json_str
                self._log(f"Generation complete ({len(json_str)} chars)", "success")
                return json_str
            except Exception as e:
                err_str = str(e)
                is_retryable = any(k in err_str for k in ("502", "503", "429", "Connection error", "upstream", "timeout", "Timeout"))
                if is_retryable and attempt < max_api_retries:
                    wait = min(10 * attempt, 60)
                    self._log(f"API error (attempt {attempt}/{max_api_retries}): {e} - retrying in {wait}s", "warning")
                    _time.sleep(wait)
                    continue
                self._log(f"Generation failed: {e}", "error")
                import traceback
                traceback.print_exc()
                return None

    def _validate(self) -> ValidationResult:
        """Validate output"""
        self._log("Validating output...")

        try:
            self.parsed_json = json.loads(self.current_output)
        except json.JSONDecodeError as e:
            return ValidationResult(is_valid=False, errors=[f"JSON parse error: {e}"])

        result = validate_stage1_schema(self.parsed_json)

        if result.is_valid:
            self._log("Validation passed", "success")
        else:
            self._log(f"Validation failed: {len(result.errors)} errors", "error")

        return result

    def _generate_fix_instructions(self, validation_result: ValidationResult) -> str:
        """Generate fix instructions"""
        errors_text = "\n".join(f"- {e}" for e in validation_result.errors)
        snippet = self.current_output[:2000] + "..." if len(self.current_output) > 2000 else self.current_output

        fix_prompt = self.fix_template.format(errors=errors_text, current_json=snippet)

        try:
            response = self.llm.invoke([
                SystemMessage(content="You are a JSON Schema fix expert"),
                HumanMessage(content=fix_prompt)
            ])
            return response.content.strip()
        except:
            return errors_text

    def _save_to_memory(self, success: bool = True):
        """Save to Memory"""
        if not self.parsed_json:
            return

        data = self.parsed_json
        zones = data.get("decoupled_zones", [])
        zone_names = [z.get("zone_name", "unknown") for z in zones]
        major = sum(1 for z in zones for obj in z.get("object_hierarchy", []) if obj.get("category") == "major")
        minor = sum(1 for z in zones for obj in z.get("object_hierarchy", []) if obj.get("category") == "minor")

        title = f"{len(zones)} zones, {major+minor} objects"

        self.memory.add(
            stage="stage1",
            type="result",
            content=self.parsed_json,
            metadata={
                "success": success,
                "title": title,
                "summary": f"{len(zones)} zones, {major} major objects, {minor} minor objects",
                "iterations": self.iteration,
                "image_path": self.image_path
            },
            tags=["stage1", "success" if success else "partial"]
        )
        self._log(f"Saved to Memory: {title}", "success")

    def _save_files(self):
        """Save to file"""
        os.makedirs(self.output_dir, exist_ok=True)

        if self.parsed_json:
            path = os.path.join(self.output_dir, "stage1_output.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.parsed_json, f, ensure_ascii=False, indent=2)
            self._log(f"JSON: {path}")

        if self.current_output:
            path = os.path.join(self.output_dir, "stage1_raw.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.current_output)
            self._log(f"Raw: {path}")

    def run(self) -> tuple:
        """
        Run Stage1.

        Returns:
            (success, result_json, metadata)
        """
        print("\n" + "=" * 60)
        print("Stage1 - Spatial semantic analysis")
        print("=" * 60)

        if not os.path.exists(self.image_path):
            self._log(f"Image does not exist: {self.image_path}", "error")
            return False, None, {}

        self._log(f"Image: {self.image_path}")

        fix_instructions = None

        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'-' * 40}")
            self._log(f"Iteration {self.iteration}/{self.max_iterations}", "step")
            print(f"{'-' * 40}")

            output = self._generate(fix_instructions)
            if not output:
                continue

            validation = self._validate()

            if validation.is_valid:
                self._save_to_memory(True)
                self._save_files()

                print("\n" + "=" * 60)
                print("Stage1 done!")
                print("=" * 60)

                return True, self.parsed_json, {"iterations": self.iteration}

            if self.iteration < self.max_iterations:
                fix_instructions = self._generate_fix_instructions(validation)

        # Hit max iterations
        self._log("Max iterations reached", "warning")
        if self.parsed_json:
            self._save_to_memory(False)
            self._save_files()

        return False, self.parsed_json, {"iterations": self.iteration}


def main():
    parser = argparse.ArgumentParser(description="Stage1 - spatial semantic analysis")
    parser.add_argument("--image", "-i", required=True, help="Image path")
    parser.add_argument("--output-dir", "-o", default="./output", help="Output directory")
    parser.add_argument("--max-iter", "-n", type=int, default=3, help="Max iterations")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    runner = Stage1Runner(
        image_path=args.image,
        output_dir=args.output_dir,
        max_iterations=args.max_iter,
        verbose=not args.quiet
    )

    success, result, meta = runner.run()

    if result:
        zones = len(result.get("decoupled_zones", []))
        objects = sum(len(z.get("object_hierarchy", [])) for z in result.get("decoupled_zones", []))
        print(f"\nResult: {'success' if success else 'partial success'}")
        print(f"   zones: {zones}, objects: {objects}, iterations: {meta['iterations']}")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
