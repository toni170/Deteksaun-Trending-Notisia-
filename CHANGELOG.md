# Mudansa — Versaun 4

**Fluxu servisu foun iha painel (la presiza terminál)**
- Tab **📥 1. Pedidu Dadus**: pedidu manuál no pedidu automátiku (padraun kada 30 minutu). Kada pedidu foti notísia to'o CSV 200 MB, rai iha lote rasik ho **CSV, JSON no SQLite**; lote la hamutuk no la repete notísia.
- Tab **🧪 2. Preprosesamentu**: importa, hamoos, case folding, tokenizasaun, hasai stopword, stemming, kategoria referénsia; rai dadus bersih (CSV, JSON, SQLite). Bele hili lote ka karga ficheiru.
- Tab **🧠 3. Treinu Modelu**: TF-IDF + K-Means, avaliasaun (akurasia, presizaun, recall, F1-score), matriz konfuzaun, tópiku, tópiku populár, tendénsia, lista liafuan.
- Servisu hala'o iha thread background; pedidu automátiku kontinua maski tab browser taka.
- Moduló foun: `src/collection`, `src/storage`, `src/jobs`, `src/preprocessing/staged_pipeline.py`, `src/training/dashboard_training.py`, `src/training/evaluation.py`, `src/training/analysis.py`, `src/dashboard/workflow_views.py`.
- Botaun antigu "Kolete Dadus Agora", "Kolete istória" no auto-refresh iha sidebar troka ho tab 1. `main.py` (terminál) sei bele uza.
- Preprosesamentu foun mantein karakter ho asentu (saúde, polítika), hasai entidade HTML no URL.
- Dependénsia: `streamlit>=1.37`, `openpyxl`.

# Mudansa — Versaun 3

- Database SQLite (`database/`): `database.py`, `models.py`, `database.db` (articles, training_runs, topics).
- `models/training.py` + `models/model.pkl`: latih/simpan/muat model topik, `predict_topic()`.
- `main.py`: langkah [6/6] rai ba database no model.pkl; opsaun `--no-db`.
- Teste foun: `tests/test_database.py`, `tests/test_training.py`. Dependénsia foun: SQLAlchemy>=2.0.

# Mudansa — Versaun 2

**Koleta**
- Koleta notísia istóriku (`--since 2016-01-01`) liuhusi WordPress REST API; hahú husi notísia tuan.
- `--since auto`: kontinua husi notísia ikus ne'ebé rai ona (update lailais).
- `--source Tatoli`: hili orijen espesífiku. Se internet para iha klaran, dadus rai hela.

**Dashboard**
- Filtru tempu: loron ida / intervalu / 2016 to agora / 7-30 loron ikus. Uza oras Dili (UTC+9).
- Tab foun **📅 Istória**: notísia kada loron/semana/fulan/tinan, rezumu kada tinan,
  tópiku kada tinan, liafuan prinsipál kada tinan.
- Tab Notísia: link klik, "Lee notísia" (konteúdu + imajen), download CSV (loke iha Excel).
- Tendénsia tópiku: agrupa otomátiku (loron/semana/fulan/tinan) — la neon bainhira dadus 10 tinan.
- Kárregamentu dadus ho cache (lalais liu bainhira ficheiru boot).

**Korresaun**
- Erru memori iha clustering ho dadus 16 rihun+ (silhouette ho sampel).
- Valór mamuk (NaN) la hamate tan dashboard.

**Atalhu Windows**: `1_install.bat`, `2_koleta_istoria_tatoli.bat`, `3_update_notisia_foun.bat`, `4_loke_dashboard.bat`.
