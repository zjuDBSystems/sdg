# SDG System

Synthetic Data Generator System

# Get Started

1. Install the required packages `uv sync`
> Using [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) as package management tool
2. Run the server `uvicorn sdg.main:app --reload`
> Follow the [tutorial](https://fastapi.tiangolo.com/tutorial/debugging/#run-your-code-with-your-debugger) to run the server with vscode debugger, the .vscode/launch.json file looks like
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        
        {
            "name": "Python Debugger: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "sdg.main:app",
                "--reload"
            ],
            "jinja": true
        }
    ]
}
```
3. Server hosted at http://127.0.0.1:8000 by default