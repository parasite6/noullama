# Noullama Python client

Ask a model loaded in Noullama questions from your own Python code.

## Install

Python 3.10 or newer. It has no dependencies.

```sh
python -m pip install noullama
```

Using `python -m pip` makes sure the package goes into the same Python that runs your script.

## Quick start

Open Noullama and load a model first.

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
print(answer.value)          # billing
print(answer.probabilities)  # {'billing': 0.98, 'technical': 0.02}
print(answer.confidence)     # 0.98
print(decision.nl_time)      # 0.025  (seconds the whole decision took)
```

- **`model`:** the model's name or alias as shown in the app's Local API tab. Use `any-available` for any free model.
- **`state`:** the text (or dict or list) you're asking about.
- **`questions`:** each question gets a name you choose (`"route"` above). You use that name to read its answer.
- **`decision`, `answer` and `nl`** are ordinary variable names. Call them whatever you like.

## Questions

There are three types. Every question needs `instructions`, which can't be empty. Descriptions are optional: use `None` for no description. An empty one like `technical=""` is an error, so a forgotten description doesn't slip through.

**`Choice`: pick one option.** Each option is a name, plus an optional description that helps the model understand it. You get back the name.

```python
Choice(instructions="Which team?", billing="Charges and refunds", technical="App errors")
Choice(instructions="Which team?", billing=None, technical=None)   # no descriptions
```

**`Score`: pick a level on a scale.** List the levels lowest first, 2 to 10 of them. You get back the level's number, counting from 0.

```python
Score(instructions="How urgent?", levels=["Can wait", "This week", "Today", "Immediately"])
```

**`Noul`: yes or no.** You get back the chance that the answer is yes, from 0 to 1.

```python
Noul(instructions="Is this a refund request?")
Noul(instructions="Is this a refund request?", true="asks for money back")  # optional descriptions
```

## Answers

Every answer has the same three fields:

| Field | Choice | Score | Noul |
| --- | --- | --- | --- |
| `.value` | the picked option, e.g. `"billing"` | the picked level, e.g. `2.0` | chance of yes, e.g. `0.93` |
| `.probabilities` | chance per option | chance per level (`"0"`, `"1"`, …) | `{"false": 0.07, "true": 0.93}` |
| `.confidence` | how sure the model is, 0 to 1 | same | same |

Native models only send the yes chance for `Noul`, so the client works out `.probabilities` and `.confidence` from it.

Native and Winnow models count `Score` and `confidence` differently. Native models give the most likely level and its chance. Winnow models give a weighted average level and an entropy-based confidence. The client shows what the model sends.

The decision as a whole has these:

- **`decision.nl_time`:** seconds from sending the request to getting the full reply. All questions share one request, so there's one time per decision.
- **`decision.model` and `decision.usage`:** the model name and token counts.
- **`decision.raw` and `answer.raw`:** the untouched reply, for anything else, such as a Score's `legend`.

## Settings and the API key

`Client()` finds the app's address and your API key, then waits. It sends nothing until you call it. It checks these places in order and uses the first value it finds:

1. Arguments: `Client(base_url="http://127.0.0.1:11440", api_key="da-noul-...")`
2. Environment variables `NOULLAMA_URL` and `NOULLAMA_API_KEY`
3. A `.env` file in the folder you run from, with the same names
4. Defaults: `http://127.0.0.1:11435` and no key

You only need a key when **Require API key** is on in the app. Keep it out of your code with a `.env` file:

```dotenv
NOULLAMA_API_KEY=da-noul-...
```

The address can be pasted from the Local API tab as is, `/v1` included. The key never appears in printouts or error messages.

Make one client and reuse it, for example in a loop:

```python
nl = Client()
for email in inbox:
    decision = nl.decide(model="d1-2B-Q8_0", state=email, questions=questions)
```

## More

- **`nl.models()`:** the model names you can use right now.
- **`nl.health()`:** whether the app is up and a model is ready.
- **`nl.wait_until_ready(model="d1-2B-Q8_0", timeout=60)`:** waits until that model is ready. Leave out `model` to wait for any model.
- **Images:** pass `images=["receipt.png"]` to `decide()`. This only works with models that have a vision projector. Winnow models are text only.

## Errors

Each problem has its own exception, so you can catch the ones you care about:

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

| Exception | Meaning |
| --- | --- |
| `NotRunning` | The app isn't open, or isn't at that address |
| `ModelUnavailable` | The model isn't loaded, is still loading, or every model is busy |
| `UnknownModel` | No model has that name (names are case-sensitive) |
| `AmbiguousModel` | The name matches more than one model |
| `BadRequest` | Something in the request is wrong. The message says what |
| `AuthError` | The API key is missing, wrong or expired |
| `Forbidden` | The API key isn't allowed to use this model |
| `InferenceFailed` | The model failed while answering |
| `NoullamaError` | The base class of all of the above; catches everything |

Every exception has `.message` (the app's own wording) and `.status` (the HTTP status code).

## Tested with

Noullama 0.5.4.

## License

The Python client is [MIT-licensed](https://github.com/Noullama/noullama/blob/sanctuary/python/LICENSE). The rest of Noullama is AGPL-3.0.
