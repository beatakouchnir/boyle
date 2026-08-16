# VHS tape: zero-config Ollama moment. Requires `boyle serve ... ` already
# running on 11434 in another terminal, and the ollama CLI installed.
# Render: vhs docs/media/ollama.tape
Output docs/media/ollama.gif
Set FontSize 15
Set Width 1000
Set Height 420
Set Theme "Catppuccin Mocha"
Set Padding 12
Set TypingSpeed 40ms

Type "ollama list"
Sleep 400ms
Enter
Sleep 1500ms
Type "ollama run qwen3-30b-a3b-instruct-2507-4bit:latest 'In one short sentence, why is the sky blue?'"
Sleep 600ms
Enter
Sleep 2800ms
Ctrl+C
Sleep 800ms
Type "# that 'ollama server' is boyle — a 30B MoE in a 20 GB budget"
Sleep 2.5s
