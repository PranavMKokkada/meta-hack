#!/usr/bin/env python3
"""
Pre-Submission Validator for OpenEnv Submission
===============================================
Checks all submission requirements before uploading to HF Spaces.

Run: python validator.py
"""

import os
import sys
import re
import yaml
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

REQUIRED_TOP_LEVEL_FILES = [
    "inference.py",
    "openenv.yaml",
    "Dockerfile",
    "requirements.txt"
]

REQUIRED_ENV_VARS = [
    "API_BASE_URL",
    "MODEL_NAME",
    "HF_TOKEN"
]

REQUIRED_OPENENV_ENDPOINTS = [
    "reset",
    "step",
    "grader"
]

REQUIRED_TASKS_MIN = 3


# ── Validators ─────────────────────────────────────────────────────────────────

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def check(condition: bool, message: str):
    """Print a check result."""
    if condition:
        print(f"  {Colors.GREEN}✅{Colors.RESET} {message}")
        return True
    else:
        print(f"  {Colors.RED}❌{Colors.RESET} {message}")
        return False


def check_file_structure():
    """Verify all required files exist."""
    print(f"\n{Colors.BOLD}1. File Structure{Colors.RESET}")
    
    all_ok = True
    for filename in REQUIRED_TOP_LEVEL_FILES:
        exists = Path(filename).exists()
        all_ok &= check(exists, f"{filename} exists in root directory")
    
    return all_ok


def check_inference_py():
    """Validate inference.py compliance."""
    print(f"\n{Colors.BOLD}2. inference.py Compliance{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("inference.py", "r") as f:
            content = f.read()
        
        # Check: Uses OpenAI Client
        all_ok &= check(
            "from openai import OpenAI" in content,
            "Uses OpenAI client (from openai import OpenAI)"
        )
        
        # Check: Gets API_BASE_URL
        all_ok &= check(
            'API_BASE_URL' in content and 'os.getenv("API_BASE_URL"' in content,
            "Reads API_BASE_URL from environment"
        )
        
        # Check: Gets MODEL_NAME
        all_ok &= check(
            'MODEL_NAME' in content and 'os.getenv("MODEL_NAME"' in content,
            "Reads MODEL_NAME from environment"
        )
        
        # Check: Gets HF_TOKEN
        all_ok &= check(
            'HF_TOKEN' in content and ('os.getenv("HF_TOKEN"' in content or 'API_KEY' in content),
            "Reads HF_TOKEN from environment"
        )
        
        # Check: Has [START] logging
        all_ok &= check(
            '[START]' in content,
            "Emits [START] structured logs"
        )
        
        # Check: Has [STEP] logging
        all_ok &= check(
            '[STEP]' in content,
            "Emits [STEP] structured logs"
        )
        
        # Check: Has [END] logging
        all_ok &= check(
            '[END]' in content,
            "Emits [END] structured logs"
        )
        
        # Check: OpenAI client creation
        all_ok &= check(
            'OpenAI(' in content and 'api_key=' in content,
            "Creates OpenAI client with api_key"
        )
        
        # Check: Uses chat.completions.create
        all_ok &= check(
            'chat.completions.create' in content,
            "Uses OpenAI chat.completions.create for LLM calls"
        )
        
    except FileNotFoundError:
        check(False, "inference.py not found")
        return False
    except Exception as e:
        check(False, f"Error reading inference.py: {e}")
        return False
    
    return all_ok


def check_openenv_yaml():
    """Validate openenv.yaml structure."""
    print(f"\n{Colors.BOLD}3. OpenEnv Specification{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("openenv.yaml", "r") as f:
            spec = yaml.safe_load(f)
        
        # Check: spec_version
        all_ok &= check(
            spec.get("spec_version") == 1,
            "spec_version set to 1"
        )
        
        # Check: name
        all_ok &= check(
            "name" in spec and spec["name"],
            "Environment has a name"
        )
        
        # Check: endpoints section exists
        endpoints = spec.get("env", {}).get("endpoints", {})
        all_ok &= check(
            len(endpoints) > 0,
            "Endpoints section exists"
        )
        
        # Check: required endpoints
        for endpoint in REQUIRED_OPENENV_ENDPOINTS:
            all_ok &= check(
                endpoint in endpoints,
                f"Has {endpoint} endpoint"
            )
        
        # Check: tasks section
        tasks = spec.get("tasks", [])
        all_ok &= check(
            len(tasks) >= REQUIRED_TASKS_MIN,
            f"Has {len(tasks)} tasks (minimum {REQUIRED_TASKS_MIN})"
        )
        
        # Check: each task has required fields
        for task in tasks:
            task_id = task.get("id", "unknown")
            has_id = "id" in task
            has_name = "name" in task
            has_difficulty = "difficulty" in task
            
            if has_id and has_name and has_difficulty:
                check(True, f"Task {task_id} is well-formed")
            else:
                check(False, f"Task {task_id} missing fields")
                all_ok = False
        
    except FileNotFoundError:
        check(False, "openenv.yaml not found")
        return False
    except yaml.YAMLError as e:
        check(False, f"openenv.yaml is not valid YAML: {e}")
        return False
    except Exception as e:
        check(False, f"Error reading openenv.yaml: {e}")
        return False
    
    return all_ok


def check_dockerfile():
    """Validate Dockerfile."""
    print(f"\n{Colors.BOLD}4. Docker Configuration{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("Dockerfile", "r") as f:
            content = f.read()
        
        all_ok &= check(
            "FROM" in content,
            "Dockerfile has FROM statement"
        )
        
        all_ok &= check(
            "python" in content.lower(),
            "Uses Python base image"
        )
        
        all_ok &= check(
            "WORKDIR" in content,
            "Sets working directory"
        )
        
        all_ok &= check(
            "requirements.txt" in content,
            "Installs requirements.txt"
        )
        
        all_ok &= check(
            "7860" in content or "PORT" in content,
            "Configures port 7860 for HF Spaces"
        )
        
    except FileNotFoundError:
        check(False, "Dockerfile not found")
        return False
    except Exception as e:
        check(False, f"Error reading Dockerfile: {e}")
        return False
    
    return all_ok


def check_requirements():
    """Check requirements.txt for essential packages."""
    print(f"\n{Colors.BOLD}5. Python Dependencies{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read().lower()
        
        all_ok &= check(
            "openai" in content,
            "requirements.txt includes 'openai' package"
        )
        
        all_ok &= check(
            "fastapi" in content or "flask" in content,
            "requirements.txt includes web framework (FastAPI/Flask)"
        )
        
        all_ok &= check(
            "pydantic" in content or "dataclass" in content,
            "requirements.txt includes data validation"
        )
        
    except FileNotFoundError:
        check(False, "requirements.txt not found")
        return False
    except Exception as e:
        check(False, f"Error reading requirements.txt: {e}")
        return False
    
    return all_ok


def check_environment_vars():
    """Check that env vars are NOT hardcoded."""
    print(f"\n{Colors.BOLD}6. Environment Variables{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("inference.py", "r") as f:
            content = f.read()
        
        for var in REQUIRED_ENV_VARS:
            all_ok &= check(
                'os.getenv' in content and var in content,
                f"Uses os.getenv() for {var} (not hardcoded)"
            )
        
    except Exception as e:
        check(False, f"Error checking env vars: {e}")
        return False
    
    return all_ok


def check_logging_format():
    """Validate [START], [STEP], [END] format."""
    print(f"\n{Colors.BOLD}7. Structured Logging Format{Colors.RESET}")
    
    all_ok = True
    
    try:
        with open("inference.py", "r") as f:
            content = f.read()
        
        # Check for [START]
        start_pattern = r'\[START\].*task=.*env=.*model='
        all_ok &= check(
            re.search(start_pattern, content) is not None,
            "[START] line includes: task=, env=, model="
        )
        
        # Check for [STEP]
        step_pattern = r'\[STEP\].*step=.*action=.*reward=.*done=.*error='
        all_ok &= check(
            re.search(step_pattern, content) is not None,
            "[STEP] line includes: step=, action=, reward=, done=, error="
        )
        
        # Check for [END]
        end_pattern = r'\[END\].*success=.*steps=.*rewards='
        all_ok &= check(
            re.search(end_pattern, content) is not None,
            "[END] line includes: success=, steps=, rewards="
        )
        
        # Check reward formatting (2 decimals)
        all_ok &= check(
            ':.2f' in content or '{:.2f}' in content,
            "Formats rewards to 2 decimal places (e.g., 0.50)"
        )
        
    except Exception as e:
        check(False, f"Error checking logging: {e}")
        return False
    
    return all_ok


def check_imports():
    """Verify no import errors."""
    print(f"\n{Colors.BOLD}8. Import Validation{Colors.RESET}")
    
    all_ok = True
    
    try:
        # Try to parse inference.py as Python
        with open("inference.py", "r") as f:
            content = f.read()
        
        compile(content, "inference.py", "exec")
        all_ok &= check(True, "inference.py has valid Python syntax")
        
    except SyntaxError as e:
        check(False, f"Syntax error in inference.py: {e}")
        return False
    except Exception as e:
        check(False, f"Error validating Python: {e}")
        return False
    
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """Run all validators."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 Pre-Submission Validator{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")
    
    results = {
        "File Structure": check_file_structure(),
        "inference.py": check_inference_py(),
        "OpenEnv Spec": check_openenv_yaml(),
        "Dockerfile": check_dockerfile(),
        "Dependencies": check_requirements(),
        "Environment Vars": check_environment_vars(),
        "Logging Format": check_logging_format(),
        "Imports": check_imports(),
    }
    
    # Summary
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.RESET}\n")
    print(f"\n{Colors.BOLD}📊 Summary{Colors.RESET}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for category, result in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if result else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"  {status} {category}")
    
    print(f"\n{Colors.BOLD}Total:{Colors.RESET} {passed}/{total} checks passed")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Submission Ready!{Colors.RESET}")
        print(f"   All checks passed. Ready to push to HF Spaces & submit.")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Fix Failures Before Submitting{Colors.RESET}")
        print(f"   {total - passed} check(s) failed. Please fix and re-run validator.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
