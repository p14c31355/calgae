use calgae::agent::{run_agent, run_interactive_loop};
use calgae::cli::Args;
use clap::Parser;
use std::io::{self, stdout};
use std::path::PathBuf;
use std::sync::Arc;
use anyhow::Result as AnyhowResult;
use crossterm::event::{self, DisableMouseCapture, EnableMouseCapture, Event as CrosstermEvent, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::cursor;
use crossterm::execute;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::text::Line;

struct AlgaeBlock;

impl Widget for AlgaeBlock {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let text_lines = vec![
            Line::raw("█▀▀█▀▀█ ▀   █▀▀█▀█▄ ▀█▀▀█▀█▄▀█▀▀█▀"),
            Line::raw("█▄▄█  █ ▀   █▄▄█  █ ▀█   █ ▀   █"),
            Line::raw("▀▀▀█▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀"),
        ];
        let paragraph = Paragraph::new(text_lines)
            .block(Block::default()
                .title("Calgae")
                .borders(Borders::ALL)
                .border_style(Style::new().fg(Color::Green)));
        paragraph.render(area, buf);
    }
}

async fn tui_app(agent: Arc<Agent>, tokens: usize, execute: bool) -> AnyhowResult<()> {
    enable_raw_mode()?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    execute!(terminal.backend_mut(), EnterAlternateScreen, DisableMouseCapture)?;
    terminal.clear()?;

    let mut messages = vec![Line::from("█▀▀█▀▀█ ▀   █▀▀█▀█▄ ▀█▀▀█▀█▄▀█▀▀█▀"), Line::from("█▄▄█  █ ▀   █▄▄█  █ ▀█   █ ▀   █"), Line::from("▀▀▀█▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀ ▀▀▀▀"), Line::from("Calgae: Lightweight LLM Runtime")];
    let mut input = String::new();
    let mut should_quit = false;

    while !should_quit {
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints([
                    Constraint::Length(4),
                    Constraint::Min(5),
                    Constraint::Length(3),
                ])
                .split(f.area());

            // Header
            let header = AlgaeBlock;
            f.render_widget(header, chunks[0]);

            // Messages
            let messages_block = Paragraph::new(messages.clone())
                .block(Block::default()
                    .title("Conversation")
                    .borders(Borders::ALL));
            f.render_widget(messages_block, chunks[1]);

            // Input
            let input_text = format!("Input: {}", input);
            let input_block = Paragraph::new(Line::raw(input_text))
                .block(Block::default()
                    .title("Type your task")
                    .borders(Borders::ALL));
            f.render_widget(input_block, chunks[2]);
            f.set_cursor(chunks[2].x + input.len() as u16 + 8, chunks[2].y + 1); // Set cursor in input
        })?;

        if let CrosstermEvent::Key(key) = crossterm::event::read()? {
            match key.code {
                KeyCode::Enter => {
                    if !input.trim().is_empty() {
                        let user_input = input.trim().to_string();
                        let response = agent.generate_code(&user_input, tokens).await?;
                        messages.push(Line::from(format!("User: {}", user_input)));
                        messages.push(Line::from(format!("Calgae: {}", response)));
                        input.clear();
                    }
                }
                KeyCode::Char(c) => input.push(c),
                KeyCode::Backspace => { input.pop(); }
                KeyCode::Esc => should_quit = true,
                _ => {}
            }
        }
    }

    execute!(terminal.backend_mut(), LeaveAlternateScreen, EnableMouseCapture)?;
    disable_raw_mode()?;
    Ok(())
}

#[tokio::main]
async fn main() -> AnyhowResult<()> {
    let args = Args::parse();

    println!("Model path: {:?}", args.model.display());

    if !args.model.exists() {
        eprintln!(
            "Warning: Model file not found at {:?}. Please download the model.",
            args.model
        );
        return Ok(());
    }

    let quantize_bits = if args.quantize { 
        Some(
            match args.quantize_mode.as_str() {
                "awq" => 4,
                "smoothquant" => 8,
                _ => 8,
            }
        ) 
    } else { 
        None 
    };

    run_agent(
        args.model,
        args.prompt,
        args.tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.execute,
        args.interactive,
        quantize_bits,
        Some(args.quantize_mode),
    )
    .await?;

    Ok(())
}
