#!/usr/bin/env python3
"""
Learning Loop for Technical Support Agent
==========================================
Evaluate → Analyze → Improve → Re-evaluate cycle

Runs inference multiple times, analyzes failures, and improves the system prompt
based on patterns observed in low-scoring tickets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import requests

# ── Configuration ─────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-8B-Instruct")
ENV_API_URL = os.getenv("ENV_API_URL", "http://localhost:7860")

TASK_IDS = ["task_easy", "task_medium", "task_hard", "task_expert"]
LEARNING_ITERATIONS = 3
SCORE_THRESHOLD = 0.5  # Below this is a failure


@dataclass
class IterationResult:
    """Results from one learning iteration."""
    iteration: int
    scores: dict[str, float]
    avg_score: float
    failed_categories: list[str]
    failed_count: int
    prompt_version: str


class LearningLoop:
    """Manages iterative learning and improvement."""

    def __init__(self):
        self.iteration_history: list[IterationResult] = []
        self.prompt_improvements: dict[int, str] = {}
        self.best_result: Optional[IterationResult] = None
        self.failing_patterns: dict[str, int] = {}

    def run_inference(self, task_id: str) -> dict:
        """Run inference for a single task."""
        resp = requests.post(f"{ENV_API_URL}/reset", json={"task_id": task_id})
        resp.raise_for_status()
        
        # Run the actual inference script subprocess
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            print(f"  ❌ Inference failed: {result.stderr}", file=sys.stderr)
            return {"score": 0.0}
        
        # Parse the JSON output (last line)
        try:
            lines = result.stdout.strip().split("\n")
            json_line = [l for l in lines if l.startswith("{")][-1]
            return json.loads(json_line)
        except (json.JSONDecodeError, IndexError):
            return {"score": 0.0}

    def run_iteration(self, iteration: int) -> IterationResult:
        """Run one complete iteration: reset → evaluate → analyze."""
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{LEARNING_ITERATIONS}")
        print(f"{'='*60}")

        scores = {}
        failed_categories = []

        for task_id in TASK_IDS:
            print(f"  Evaluating {task_id}...", end=" ")
            result = self.run_inference(task_id)
            score = result.get("score", 0.0)
            scores[task_id] = score
            
            if score < SCORE_THRESHOLD:
                failed_categories.append(task_id)
                self.failing_patterns[task_id] = self.failing_patterns.get(task_id, 0) + 1
                print(f"❌ FAIL ({score:.2f})")
            else:
                print(f"✅ PASS ({score:.2f})")

        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        result = IterationResult(
            iteration=iteration,
            scores=scores,
            avg_score=avg_score,
            failed_categories=failed_categories,
            failed_count=len(failed_categories),
            prompt_version=f"v{iteration}"
        )
        
        self.iteration_history.append(result)
        
        if not self.best_result or avg_score > self.best_result.avg_score:
            self.best_result = result
        
        return result

    def analyze_failures(self, result: IterationResult) -> dict[str, str]:
        """Analyze failure patterns and suggest improvements."""
        improvements = {}
        
        if result.failed_count == 0:
            print("  ✅ No failures! Learning complete.")
            return improvements
        
        print(f"\n  📊 Analysis:")
        print(f"     - Failed {result.failed_count}/{len(TASK_IDS)} tasks")
        print(f"     - Avg score: {result.avg_score:.2f}")
        
        # Pattern detection
        if "task_hard" in result.failed_categories:
            improvements["complex_tickets"] = "Add guidance for multi-issue tickets"
        
        if "task_expert" in result.failed_categories:
            improvements["escalation"] = "Improve escalation detection"
        
        if any("easy" in cat for cat in result.failed_categories):
            improvements["accuracy"] = "Improve classification accuracy"
        
        return improvements

    def improve_system_prompt(self, failures: dict[str, str]) -> str:
        """Generate improved system prompt based on failure analysis."""
        if not failures:
            return ""
        
        print(f"\n  🔧 Improvements to apply:")
        for key, improvement in failures.items():
            print(f"     - {improvement}")
        
        # Build improvement instructions
        improvements_text = "\n".join(f"- {v}" for v in failures.values())
        
        enhancement = f"""
ENHANCEMENT (Applied after iteration):
{improvements_text}

Remember: Pay extra attention to {', '.join(failures.keys())} cases.
Verify your JSON is always valid and complete.
"""
        return enhancement

    def run_learning_loop(self) -> dict:
        """Execute the complete learning loop."""
        print(f"\n🚀 Starting learning loop: {LEARNING_ITERATIONS} iterations")
        print(f"   Model: {MODEL_NAME}")
        print(f"   Tasks: {', '.join(TASK_IDS)}")

        for i in range(1, LEARNING_ITERATIONS + 1):
            # 1. Run evaluation
            result = self.run_iteration(i)
            
            # 2. Analyze failures
            failures = self.analyze_failures(result)
            
            # 3. Improve prompt (for next iteration)
            if i < LEARNING_ITERATIONS and failures:
                improvement = self.improve_system_prompt(failures)
                self.prompt_improvements[i] = improvement
                print(f"   → Ready for iteration {i+1}")

        # Summary
        return self.generate_summary()

    def generate_summary(self) -> dict:
        """Generate final summary and recommendations."""
        summary = {
            "total_iterations": len(self.iteration_history),
            "best_iteration": self.best_result.iteration if self.best_result else None,
            "best_avg_score": self.best_result.avg_score if self.best_result else 0.0,
            "best_scores": self.best_result.scores if self.best_result else {},
            "all_iterations": [
                {
                    "iteration": r.iteration,
                    "avg_score": r.avg_score,
                    "scores": r.scores,
                    "failed_count": r.failed_count,
                }
                for r in self.iteration_history
            ],
            "failing_patterns": self.failing_patterns,
            "recommendations": self.generate_recommendations(),
        }
        return summary

    def generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on patterns."""
        recommendations = []
        
        if self.best_result and self.best_result.avg_score < 0.7:
            recommendations.append("⚠️  Overall score is low. Consider training on more examples.")
        
        if self.failing_patterns:
            worst_task = max(self.failing_patterns, key=self.failing_patterns.get)
            recommendations.append(f"🎯 Focus improvement efforts on {worst_task}.")
        
        # Check for improvement across iterations
        if len(self.iteration_history) > 1:
            first_score = self.iteration_history[0].avg_score
            last_score = self.iteration_history[-1].avg_score
            improvement = last_score - first_score
            
            if improvement > 0.05:
                recommendations.append(f"✅ Learning loop effective! Improvement: +{improvement:.2f}")
            elif improvement > 0:
                recommendations.append(f"📈 Small improvement detected: +{improvement:.2f}")
            else:
                recommendations.append("⚠️  No improvement detected. Try different improvement strategies.")
        
        return recommendations


def main():
    """Run the learning loop."""
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    print("🧠 Technical Support Agent Learning Loop")
    print("=========================================\n")

    loop = LearningLoop()
    summary = loop.run_learning_loop()

    # Print final summary
    print(f"\n\n{'='*60}")
    print("📊 LEARNING LOOP SUMMARY")
    print(f"{'='*60}")
    print(f"\nBest Iteration: #{summary['best_iteration']}")
    print(f"Best Average Score: {summary['best_avg_score']:.2f}")
    print(f"\nBest Scores by Task:")
    for task_id, score in summary['best_scores'].items():
        status = "✅" if score >= SCORE_THRESHOLD else "❌"
        print(f"  {status} {task_id}: {score:.2f}")

    print(f"\nFailing Patterns (across all iterations):")
    for task_id, count in summary['failing_patterns'].items():
        print(f"  - {task_id}: failed {count} time(s)")

    print(f"\n🎯 Recommendations:")
    for rec in summary['recommendations']:
        print(f"  {rec}")

    # Output machine-readable summary
    print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
