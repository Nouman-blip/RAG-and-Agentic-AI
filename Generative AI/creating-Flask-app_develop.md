# Cheat Sheet: Web App Deployment Using Flask

> **Estimated time needed: 5 minutes**

---

## 📦 Quick Reference Table

| Package / Method | Description |
|---|---|
| `Flask` | Used to instantiate an object of the Flask class named `app` |
| `@app.route` decorator | Maps URLs to specific functions in a Flask application |
| `200 OK` status | Automatically returned on success, also default for `jsonify()` |
| `Error 4xx` | Client-side errors (bad request, unauthorized, not found, etc.) |
| `Error 500` | Server-side error |

---

## 🔧 Flask — App Instantiation

Used to create an object of the Flask class.

```python
from flask import Flask

app = Flask(__name__)
```

---

## 🔗 `@app.route` Decorator

Maps a URL to a specific function in your Flask application.

```python
@app.route('/')
def hello_world():
    return "My first Flask application in action!"
```

---

## ✅ 200 OK Status

Flask servers automatically return a `200 OK` status when you return from an `@app.route` method. `200` is also the default when using `jsonify()`. You can also return it explicitly:

```python
@app.route('/')
def hello_world():
    return ("My first Flask application in action!", 200)
```

---

## ⚠️ Error 4xx — Client-Side Errors

| Code | Meaning |
|---|---|
| `400` | Invalid request — parameters missing or malformed |
| `401` | Credentials missing or invalid |
| `403` | Client credentials insufficient to fulfill the request |
| `404` | Resource not found on the server |
| `405` | Requested operation not supported |
| `422` | Unprocessable entity — input parameter missing or invalid |

```python
@app.route('/')
def search_response():
    query = request.args.get("q")

    if not query:
        return {"error_message": "Input parameter missing"}, 422

    # Fetch the resource from the database
    resource = fetch_from_database(query)

    if resource:
        return {"message": resource}
    else:
        return {"error_message": "Resource not found"}, 404
```

---

## 🔥 Error 500 — Server-Side Error

Used when an unexpected error occurs on the server.

```python
@app.errorhandler(500)
def server_error(error):
    return {"message": "Something went wrong on the server"}, 500
```

---

## 🗺️ Status Code Summary

```
2xx  →  Success        (200 OK)
4xx  →  Client Error   (400 Bad Request, 401 Unauthorized,
                        403 Forbidden, 404 Not Found, 405 Not Allowed)
5xx  →  Server Error   (500 Internal Server Error)
```

---

> 💡 **Tip:** Always handle both client and server errors explicitly in production Flask apps to provide meaningful feedback and avoid exposing internal details.
