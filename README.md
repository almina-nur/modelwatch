# ModelWatch

ModelWatch; veri kaymasi (data drift) analizi, MLflow kayitli model izleme, seviyelendirilmis uyari uretimi, SQLite gecmis kaydi ve FastAPI tahmin/izleme servisi sunan bir MLOps prototipidir.

> **Veri Kaynagi Notu:** Varsayilan `src/data_loader.py` akisi `sklearn.datasets.make_classification` tabanli **sentetik veri simulasyonu** kullanir. Dis/gercek kayitlar `/monitor/run` isteginde `records` alani uzerinden gonderilebilir.

## Ortam Degiskenleri

| Degisken | Varsayilan | Aciklama |
| :--- | :--- | :--- |
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking ve registry URI'si |
| `MODELWATCH_MODEL_NAME` | `ModelWatch-CustomerSatisfaction` | Kayitli MLflow model adi |
| `MODELWATCH_MODEL_ALIAS` | `staging` | Kullanilacak model alias'i |
| `MODELWATCH_HISTORY_DB` | `data/modelwatch_history.db` | SQLite gecmis veritabani yolu |
| `MODELWATCH_WEBHOOK_URL` | *(yok)* | Istege bagli alarm webhook URL'si |

## Kullanim ve Test

```bash
pip install -e ".[dev]"
python -m src.train
uvicorn api.main:app --reload
ruff check src api tests
pytest -v
```

Ornek `/monitor/run` ciktisi icin `docs/sample_monitoring_run.json` ve lisans icin [MIT Lisansi](LICENSE) dosyalarina bakabilirsiniz.
