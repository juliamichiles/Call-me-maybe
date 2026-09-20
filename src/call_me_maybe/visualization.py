from typing import Any, List, Optional, Sequence, Set

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .state_machine import JSONStateMachine

# TODO: Add if TYPE_CHECKING here
class GenVisualizer:
    """Display the constrained-decoding process using Rich."""
    def __init__(
            self,
            promt: str,
            max_tokens: int,
            vocab_size: int
    ) -> None:
        self.promt = promt
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size

        self.generated_tokens: List[str] = []
        self.generated_ids: List[int] = []
        self.allowed_count = 0
        self.current_token: Optional[str] = None
        self.current_token_id: Optional[int] = None
        self.current_state = "INITIALIZING"
        self.status = "Starting generation..."
        self.selected_function = ""
        self.current_parameter = ""
        self.current_parameter_type = ""
        self.completed = False
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

    def update_constraints(
            self,
            allowed_ids: Set[int],
            state_machine: JSONStateMachine
            ) -> None:
        """Update information about the current constraints."""
        # FIXME: Rename to just update or update_info somethink like that
        self.allowed_count = len(allowed_ids)
        self.current_state = state_machine.current_state.name
        
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
            deterministic: bool = False
    ) -> None:
        """Record a generated token."""
        self.current_token_id = token_id
        self.current_token = token_text
        self.generated_ids.append(token_id)
        self.generated_tokens.append(token_text)

        if deterministic:
            self.status = "Deterministic token emitted"
        else:
            self.status = "LLM token selected"
        self._refresh()

    def finish(
            self,
            state_machine: JSONStateMachine,
    ) -> None:
        """Mark generation as complete."""
        self.completed = True
        self.current_state = state_machine.current_state.name
        self.status = "Generation complete"
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live dashboard."""
        self.live.update(self._render())
    
    def _render(self) -> Group:
        """Build the complete dashboard."""

        return Group(
            self._render_header(),
            self._render_prompt(),
            self._render_generation(),
            self._render_state(),
            self._render_constraints(),
            self._render_last_token(),
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
                Text(self.promt),
                title="Prompt",
                border_style="blue"
        )

    def _render_generation(self) -> Panel:
        """Render the generated JSON."""
        output = "".join(self.generated_tokens)
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
        table = Table.grid(expand=True)
        table.add_column(style="bold", width=18)
        table.add_column()
        table.add_row("Vocabulary", f"{self.vocab_size:,}")
        table.add_row("Allowed", f"{self.allowed_count:,}")
        table.add_row("Masked", f"{masked:,}")

        if self.vocab_size:
            percentage = (self.allowed_count / self.vocab_size * 100)
        else:
            percentage = 0.0

        table.add_row("Allowed %", f"{percentage:.4f}%")
        
        return Panel(
                table,
                title="constraints",
                border_style="yellow"
        )
    
    def _render_last_token(self) -> Panel:
        """Render information about the latest token."""
        table = Table.grid(expand=True)
        table.add_column(style="bold", width=18)
        table.add_column()

        if self.current_token_id is None:
            table.add_row("Token", "-")
            table.add_row("Status", "Waiting")
        else:
            table.add_row("Token ID", str(self.current_token_id))
            table.add_row("Token", repr(self.current_token))
            table.add_row("Status", "✓ Selected")

        return Panel(
                table,
                title="Last Token",
                border_style="cyan"
        )
