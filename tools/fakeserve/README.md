# fakeserve: GPU-free fakes for llama-server and llama-swap

Loopback-only (127.0.0.1), stdlib-only stand-ins used by the RTR (P3) and extension (P6) tests (spec-v2 §8a, Q29a). They never start a model, touch a GPU or reach the network.

- `fake_llama_server.py`: OpenAI `/v1/chat/completions`, `/health`, `/v1/models`. Modes `auto` (pure JSON only when `response_format` is sent), `dossier`, `malformed`, `error500`, `loading`, `text`, plus a configurable delay. There are no slot endpoints (Q1a).
- `fake_llama_swap.py`: mirrors the v277 routes `/running`, `/unload`, `/api/models/unload[/<model>]`, `/v1/models`, `/upstream/<model>/...` and model-routed chat. It simulates a matrix solver (vars, `& | () +ref`, evict_costs), load and unload delays, load-failure injection, and unloads that wait for in-flight requests.

Both fakes expose `/fake/control`, `/fake/reset` and a log (`/fake/log` or `/fake/events`) for assertions; each module's docstring lists the details.

```sh
python3 tools/fakeserve/fake_llama_server.py --port 0 --mode auto
python3 tools/fakeserve/fake_llama_swap.py --config cfg.json --port 0   # JSON config, see docstring
```

In tests, use them in-process: `FakeLlamaSwap(cfg).start()` then `.port`, `.events()`, `.close()`. Self-tests are in `tests/test_fakeserve.py`.
