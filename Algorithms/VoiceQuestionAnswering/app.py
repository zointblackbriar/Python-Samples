


app = Flask(__name__)
app = Api(app)

@app.route("/")
@app.route("/index")
def index():
	return render_template("index.html")