"""Gradio demo UI for SOC AI Gym LLM training runs."""

from __future__ import annotations

from inference import run_training


def run_easy() -> str:
    logs = run_training("easy", client=None, episodes=5)
    return "\n".join(logs)


def run_medium() -> str:
    logs = run_training("medium", client=None, episodes=5)
    return "\n".join(logs)


def run_hard() -> str:
    logs = run_training("hard", client=None, episodes=5)
    return "\n".join(logs)


def build_ui():
    import gradio as gr

    theme = gr.themes.Soft(
        primary_hue="gray",
        secondary_hue="gray",
        neutral_hue="slate",
    )

    css = """
    html, body {
      background: #000000 !important;
    }
    .gradio-container {
      background: #000000 !important;
    }
    .hero-card {
      background: radial-gradient(circle at 20% 20%, #18181b 0%, #000000 70%);
      border-radius: 18px;
      padding: 20px 22px;
      border: 1px solid #2a2a2a;
      margin-bottom: 12px;
    }
    .hero-title {
      font-size: 32px;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin: 0;
      color: #f8fafc;
    }
    .hero-sub {
      margin-top: 8px;
      color: #d4d4d8;
      line-height: 1.55;
      font-size: 14px;
    }
    .chip {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #3f3f46;
      color: #f4f4f5;
      font-size: 12px;
      margin-right: 8px;
      margin-top: 8px;
    }
    .task-card {
      border: 1px solid #2a2a2a;
      border-radius: 14px;
      padding: 14px 16px;
      background: #09090b;
      min-height: 140px;
    }
    .task-title {
      font-weight: 700;
      font-size: 16px;
      margin-bottom: 6px;
      color: #fafafa;
    }
    .task-sub {
      color: #a1a1aa;
      font-size: 13px;
      line-height: 1.4;
    }
    """

    with gr.Blocks(theme=theme, css=css, title="SOC AI Gym - Judge Demo") as demo:
        gr.Markdown(
            """
<div class="hero-card">
  <h1 class="hero-title">SOC AI Gym</h1>
  <p class="hero-sub">
    SOC AI Gym is an LLM-powered cyber incident response benchmark built to evaluate how AI analysts handle realistic SOC workflows.
    Agents interact with a deterministic OpenEnv-style backend, parse alerts/logs/users, execute containment actions, and are scored with
    both step-level rewards and deterministic final grading. This judge demo runs episodic traces across three scenarios
    (brute force, suspicious login, and multi-stage attack) to show decision quality, safety, and consistency.
  </p>
  <p class="hero-sub" style="margin-top:10px;">
    <b>Core capabilities:</b> smart action defaults, multi-layer reward shaping, strict task grading, and robust HF model integration.
  </p>
  <span class="chip">OpenEnv-Compatible</span>
  <span class="chip">FastAPI + Gradio</span>
  <span class="chip">HF LLM Inference</span>
  <span class="chip">Deterministic Grader</span>
  <span class="chip">3 SOC Scenarios</span>
</div>
            """
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """
<div class="task-card">
  <div class="task-title">EASY - Brute Force Attack</div>
  <div class="task-sub">
    Detect repeated login failures and block the attacking IP.
    Ideal workflow: investigate -> block_ip.
  </div>
</div>
                    """
                )
            with gr.Column():
                gr.Markdown(
                    """
<div class="task-card">
  <div class="task-title">MEDIUM - Suspicious Login</div>
  <div class="task-sub">
    Investigate unusual login geography and disable compromised user
    after confirming suspicious behavior.
  </div>
</div>
                    """
                )
            with gr.Column():
                gr.Markdown(
                    """
<div class="task-card">
  <div class="task-title">HARD - Multi-Stage Attack</div>
  <div class="task-sub">
    Trace phishing -> account compromise -> exfiltration, then fully contain
    by blocking IP, disabling user, and stopping exfiltration.
  </div>
</div>
                    """
                )

        gr.Markdown("## Run Scenario")
        with gr.Row():
            btn1 = gr.Button("Run Easy", variant="primary")
            btn2 = gr.Button("Run Medium", variant="secondary")
            btn3 = gr.Button("Run Hard", variant="secondary")

        output = gr.Textbox(
            lines=18,
            label="Execution Logs (Episode Trace)",
            placeholder="Episode logs and reward/score traces appear here...",
            max_lines=26,
            show_copy_button=True,
        )

        btn1.click(run_easy, outputs=output)
        btn2.click(run_medium, outputs=output)
        btn3.click(run_hard, outputs=output)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
