# Toki Pona: Translator API

A basic API to translate Toki Pona to English and vice versa.

## Usage

### v1: M2M100 (multi-language)

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "mi wile toki kepeken toki pona lon tenpo ni", "src": "tl", "target": "en"}' http://localhost:8080/v1
```

### v2: GPT-2 (Toki Pona <-> English)

```bash
curl -X POST -H "Content-Type: application/json" -d '{"text": "mi wile toki kepeken toki pona lon tenpo ni", "direction": "to_english"}' http://localhost:8080/v2
```