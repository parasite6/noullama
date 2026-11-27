# Noullama Python client

## Install

Use Python 3.10 or newer. The client has no runtime dependencies.

```sh
pip install "git+https://github.com/parasite6/noullama#subdirectory=python"
```

## Quick start

Open Noullama and load a decision model. Replace `d1-2B-Q8_0` with the model name or alias shown in the Local API tab. Use `any-available` to send the request to any free, ready instance.

```python
from noullama import Choice, Client, Noul, Score

nl = Client()
decision = nl.decide(
    model="d1-2B-Q8_0",
    state="I was charged twice. Please refund the duplicate payment.",
    questions={
        "route": Choice(
            instructions="Which team should handle this?",
            billing="Charges and refunds",
            technical="App errors",
        ),
        "urgency": Score(
            instructions="How urgent is this?",
            levels=["Can wait", "This week", "Today", "Immediately"],
        ),
        "refund": Noul(instructions="Is the customer asking for a refund?"),
    },
)
answer = decision["route"]
print(answer.value)
probabilities = answer.probabilities
confidence = answer.confidence
```

## How `Client()` works

The client stores the base URL, an optional API key, and the request timeout, which defaults to 900 seconds. Creating it sends no request. It looks up the URL and key once, in this order.

1. The keyword arguments `base_url=` and `api_key=`.
2. The environment variables `NOULLAMA_URL` and `NOULLAMA_API_KEY`.
3. The same names in a `.env` file in the current directory.
4. The default URL `http://127.0.0.1:11435` and no API key.

When **Require API key** is on in Noullama, put the key in your environment or `.env`.

```dotenv
NOULLAMA_API_KEY=da-noul-...
```

You can paste the URL from the Local API tab into `NOULLAMA_URL`, including its `/v1` suffix.

```text
your script                Client                              Noullama app
nl.decide(model=...) -->   converts Choice/Noul/Score to JSON
                           adds the key if set
                           POST /v1/systemone             -->  runs the questions
                           reads the reply                 <--  sends back JSON
                           wraps each answer as
                           .value/.probabilities/.confidence
decision["route"]    <--   returns the Decision
```

Create one client before a loop and call `nl.decide(...)` on it for each item. Use `nl.health()` to check the gateway or `nl.models()` to list ready model names. Call `nl.wait_until_ready(model="d1-2B-Q8_0")` to wait for a specific name, or omit `model` to wait for any model.

If the model has an image projector, pass paths, bytes, or data URLs in `images=`. Winnow models accept text only.

## Answers

| Field | Choice | Score | Noul |
| --- | --- | --- | --- |
| `.value` | Selected option key | Gateway score | Probability of true |
| `.probabilities` | Probability by option key | Probability by level index (`"0"`, `"1"`, …) | Probability by `"false"` and `"true"` |
| `.confidence` | Gateway confidence | Gateway confidence | Gateway confidence, or the higher of the two probabilities |

For a yes/no answer with true probability `p`, the client fills in missing probabilities as `{"false": 1 - p, "true": p}` and missing confidence as `max(p, 1 - p)`. For Choice and Score, missing probabilities or confidence are `None`.

On the native model observed for this client, score is the most likely level index and confidence is the top option's probability. Winnow gives a probability-weighted score and computes confidence as `1 - entropy / ln(number of options)`. The client keeps the numbers the gateway sends.

Use `answer.raw` to read fields such as the native Score `legend`. Use `decision.raw` for the complete response, or `decision.model` and `decision.usage` for those fields directly.

`decision.nl_time` is how long the decision took, in seconds: from sending the request to reading the full response. All questions in one `decide()` call share one request, so there is one time per decision, not per answer.

```python
decision = nl.decide(...)
print(decision.nl_time)  # 0.84
```

## Errors

Gateway exceptions derive from `NoullamaError` and carry `.status` and `.message`. If a gateway error echoes the configured API key, the client replaces it with `***`.

| Status | Exception | When |
| --- | --- | --- |
| 400 | `BadRequest` | Invalid request or missing model |
| 401 | `AuthError` | Missing, unknown, or expired key |
| 403 | `Forbidden` | Key cannot access this model, or the Host or Origin check failed |
| 404 | `UnknownModel` | The gateway does not know that name |
| 409 | `AmbiguousModel` | The name reaches more than one model |
| 503 | `ModelUnavailable` | Model is unloaded or loading, no instance is available for `any-available`, or the gateway cannot read its key file |
| 502 | `InferenceFailed` | The model call failed |
| Other or plain-text HTTP error | `NoullamaError` | For example, an oversized request (413) or wrong content type (415) |
| Connection refused | `NotRunning` | The gateway is down at the configured URL |

```python
from noullama import Client, ModelUnavailable, NotRunning, Noul

try:
    decision = Client().decide(
        model="d1-2B-Q8_0",
        state="Please refund my duplicate payment.",
        questions={"refund": Noul(instructions="Is this a refund request?")},
    )
except NotRunning:
    print("Open Noullama first")
except ModelUnavailable:
    print("That model isn't loaded yet")
```

`wait_until_ready()` retries `NotRunning` and `ModelUnavailable` and raises a built-in `TimeoutError` if its deadline expires. Other gateway errors surface immediately.

## Tested against

Noullama 0.5.4 response fixtures.

## License

The Python client is [MIT-licensed](LICENSE). The rest of Noullama is AGPL-3.0.
