from flask import Flask
from flask_cors import CORS

from Controller.detailsController import details_bp
from Controller.forecastController import forecast_bp


app = Flask(__name__)
CORS(app)

# Inregistrezi modulele
app.register_blueprint(details_bp, url_prefix="/details")
app.register_blueprint(forecast_bp, url_prefix="/forecast")

if __name__ == "__main__":
    app.run(debug=True)