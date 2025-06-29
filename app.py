from flask import Flask
from flask_cors import CORS

from controller.detailsController import details_bp
from controller.forecastController import forecast_bp


app = Flask(__name__)
CORS(app)

app.register_blueprint(details_bp, url_prefix="/details")
app.register_blueprint(forecast_bp, url_prefix="/forecast")

if __name__ == "__main__":
    app.run(debug=True)