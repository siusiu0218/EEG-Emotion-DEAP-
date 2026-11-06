# S0-E01：确认实际使用的 Python

目标：区分 PowerShell 终端与 Python 解释器，确认项目命令实际调用哪个 Python。

前置知识：终端负责接收命令；Python 解释器执行 Python 代码；`-c` 表示执行后面的短代码；`sys.executable` 返回当前解释器路径。

依据：`docs/LEARNING_PLAN.md` 阶段 S0；`docs/REPRODUCTION_PROTOCOL.md` 第 7.3 节使用项目 `.venv` 的解释器。

在项目根目录的 PowerShell 中，先预测输出，再亲自运行：

```powershell
.\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
```

交付物：预测的路径、实际终端输出，以及一句话解释为什么命令明确写出 `.venv\Scripts\python.exe`。不确定的部分可以直接说明。

验收：输出指向项目解释器；能区分终端与解释器，并解释显式路径的用途。单题完成不代表整个 S0 通过；收到答案后再安排变式或补课。

限制：只检查解释器，不启动训练、不改研究代码、不改数据。若报错，提交完整报错。

状态：已出题，待用户作答；助手未代运行。
