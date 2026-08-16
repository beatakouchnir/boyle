# VHS tape: the hero shot — forecast a 400B model before downloading it.
# Render: vhs docs/media/predict.tape   (from the repo root, venv synced)
Output docs/media/predict.gif
Set FontSize 15
Set Width 1000
Set Height 340
Set Theme "Catppuccin Mocha"
Set Padding 12
Set TypingSpeed 40ms

Hide
Type "source .venv/bin/activate && clear"
Enter
Sleep 1s
Show

Type "boyle predict mlx-community/Qwen3.5-397B-A17B-4bit --budget 90GB"
Sleep 600ms
Enter
Sleep 6s
Type "# 224 GB model, forecast from headers alone — nothing downloaded"
Sleep 2.5s
