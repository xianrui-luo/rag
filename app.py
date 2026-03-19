from src.ui import build_app


app = build_app()


if __name__ == "__main__":
    app.launch(server_name="127.0.0.1")
