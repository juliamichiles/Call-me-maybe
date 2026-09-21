from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Set

from .errors import CallMeError

try:
    from rich.console import Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except (ImportError, ModuleNotFoundError) as e:
      raise CallMeError(e)

if TYPE_CHECKING:
    from .state_machine import JSONStateMachine


class GenVisualizer:
    """Display the constrained-decoding process using Rich."""
    def __init__(
            self,
            prompt: str,
            max_tokens: int,
            vocab_size: int
    ) -> None:
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        self.generated_tokens: List[str] = []
        self.generated_ids: List[int] = []
        self.current_buffer = ""
        self.allowed_count = 0
        self.current_token: Optional[str] = None
        self.current_token_id: Optional[int] = None
        self.current_state = "INITIALIZING"
        self.status = "Starting generation..."
        self.selected_function = ""
        self.current_parameter = ""
        self.current_parameter_type = ""
        self.completed = False
        
        self.timings: Dict[str, float] = {}

        self.live = Live(
                self._render(),
                refresh_per_second=10,
                transient=False
        )
    
    def start(self) -> None:
        """Start the live Rich dashboard."""
        self.live.start()

    def stop(self) -> None:
        """Stop the live Rich dashboard."""
        self.live.stop()

    def update_profiling(self, timings: Dict[str, float]) -> None:
        """Update the profiling metrics."""
        self.timings = timings

    def update_constraints(
            self,
            allowed_ids: Set[int],
            state_machine: "JSONStateMachine"
            ) -> None:
        """Update information about the current constraints."""
        self.allowed_count = len(allowed_ids)
        self.current_state = state_machine.current_state.name
        self.current_buffer = state_machine.get_full_buffer

        if state_machine.selected_function:
            self.selected_function = state_machine.selected_function.name
        else:
            self.selected_function = ""

        self.current_parameter = state_machine.current_param_name
        self.current_parameter_type = state_machine.current_param_type
        self.status = f"{self.allowed_count:,} tokens allowed"
        self._refresh()

    def update_logits(
            self,
            logits: Sequence[float]
    ) -> None:
        """Record the arrival of logits from the LLM."""
        self.status = f"LLM returned {len(logits):,} logits"
        self._refresh()

    def update_token(
            self,
            token_id: int,
            token_text: str,
            state_machine: Optional["JSONStateMachine"] = None,
            deterministic: bool = False
    ) -> None:
        """Record a generated token."""
        self.current_token_id = token_id
        self.current_token = token_text
        self.generated_ids.append(token_id)
        self.generated_tokens.append(token_text)
        if state_machine:
            self.current_buffer = state_machine.get_full_buffer

        if deterministic:
            self.status = "Deterministic token emitted"
        else:
            self.status = "LLM token selected"
        self._refresh()

    def finish(
            self,
            state_machine: "JSONStateMachine",
    ) -> None:
        """Mark generation as complete."""
        self.completed = True
        self.current_state = state_machine.current_state.name
        self.current_buffer = state_machine.get_full_buffer
        total_t = sum(self.timings.values()) if self.timings else 0
        self.status = f"Generation complete in {total_t:.2f}s"
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live dashboard."""
        self.live.update(self._render())

    def _render(self) -> Group:
        """Build the complete dashboard."""
        
        # three smaller panels side-by-side using a grid
        bottom_grid = Table.grid(expand=True, padding=(0, 1))
        bottom_grid.add_column(ratio=1) # 25% width
        bottom_grid.add_column(ratio=1) # 25% width
        bottom_grid.add_column(ratio=2) # 50% width for the charts
        
        bottom_grid.add_row(
            self._render_constraints(),
            self._render_last_token(),
            self._render_profiling()
        )
        return Group(
            self._render_header(),
            self._render_prompt(),
            self._render_generation(),
            self._render_state(),
            bottom_grid
        )

    def _render_header(self) -> Panel:
        """Render the dashboard header."""
        return Panel(
                Text(
                    "CALL ME MAYBE — CONSTRAINED DECODING",
                    style="bold cyan",
                ),
                border_style="cyan"
        )

    def _render_prompt(self) -> Panel:
        """Render the user prompt."""
        return Panel(
                Text(self.prompt),
                title="Prompt",
                border_style="blue"
        )

    def _render_generation(self) -> Panel:
        """Render the generated JSON."""
        output = self.current_buffer
        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", style="dim")
        token_count = len(self.generated_tokens)
        table.add_row(
                Text(output or "Waiting for generation..."),
                Text(f"{token_count} / {self.max_tokens}")
        )

        return Panel(
                table,
                title="Generated Output",
                border_style="green"
        )

    def _render_state(self) -> Panel:
        """Render the state-machine information."""
        table = Table.grid(expand=True)
        table.add_column(style="bold", width=18)
        table.add_column()
        table.add_row("State", self.current_state)
        table.add_row("Function", self.selected_function or "-")
        table.add_row("Parameter", self.current_parameter or "-")
        table.add_row("Parameter type", self.current_parameter_type or "-")

        return Panel(
            table,
            title="Decoder State",
            border_style="magenta",
        )

    def _render_constraints(self) -> Panel:
        """Render token constraint statistics."""
        masked = self.vocab_size - self.allowed_count
        # autosizes inside the 1/3 column
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("Vocab", f"{self.vocab_size:,}")
        table.add_row("Allowed", f"{self.allowed_count:,}")
        table.add_row("Masked", f"{masked:,}")

        if self.vocab_size:
            percentage = (self.allowed_count / self.vocab_size * 100)
        else:
            percentage = 0.0

        table.add_row("Allowed %", f"{percentage:.4f}%")
        return Panel(table, title="constraints", border_style="bright_yellow")

    def _render_last_token(self) -> Panel:
        """Render information about the latest token."""
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold")
        table.add_column(justify="right")

        if self.current_token_id is None:
            table.add_row("Token", "-")
            table.add_row("Status", "Waiting")
        else:
            table.add_row("ID", str(self.current_token_id))
            token_repr = repr(self.current_token)
            if len(token_repr) > 12:
                token_repr = token_repr[:9] + "...'"
            table.add_row("Token", token_repr)
            table.add_row("Status", "✓ Selected")

        return Panel(table, title="Last Token", border_style="cyan")

    def _render_profiling(self) -> Panel:
        """Render performance profiling charts."""
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_column(justify="left")
        
        if not self.timings:
            table.add_row("Waiting for data...", "", "")
            return Panel(
                    table, 
                    title="Performance", 
                    border_style="bright_magenta"
            )

        total_time = sum(self.timings.values())
        if total_time <= 0:
            total_time = 0.001
        colors = {
                "Encoding": "bright_blue",
                "SM Allow": "bright_yellow",
                "LLM": "bright_magenta",
                "Select": "bright_cyan",
                "SM Update": "bright_green"
        }
        
        for process, t in self.timings.items():
            pct = t/ total_time
            bar_len = 12
            filled = int(pct * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            color = colors.get(process, "white")
            table.add_row(
                    Text(process, style=color),
                    Text(f"{t:.3f}s", style=color),
                    Text(f"{bar} {pct*100:4.1f}%", style=color)
            )
        table.add_row("", "", "")
        tokens = len(self.generated_tokens)
        tps = tokens / total_time if total_time > 0 else 0 
        table.add_row(
            Text("Total Time", style="bold white"),
            Text(f"{total_time:.3f}s", style="bold white"),
            Text(f"  Speed: {tps:.1f} tok/s", style="bold dim")
        )
        return Panel(
                table, 
                title="Performance Profiling", 
                border_style="bright_magenta"
        )
