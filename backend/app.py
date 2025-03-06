from flask import Flask
from routes.detect_emotion import emotion_bp  # Import emotion detection route

app = Flask(__name__)
app.register_blueprint(emotion_bp)  # Register API route

if __name__ == "__main__":
    app.run(debug=True)
