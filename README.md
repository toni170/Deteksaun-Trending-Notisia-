# Mining Notísia Online no Deteksaun Topiku

Sistema Python atu foti notísia husi portal Timor-Leste (Tatoli, Timor Post, no seluk tan), prepara dadus,
treina modelu **TF-IDF + K-Means** no hatudu tópiku, tendénsia no lista liafuan iha **painel Streamlit**.
Fluxu hotu hala'o iha painel, la presiza terminál.

## Oinsá uza (Windows)

1. Klik 2x `1_install.bat` (dala ida de'it).
2. Klik 2x `4_loke_dashboard.bat` atu loke painel.
3. Hala'o tab 1 → 2 → 3 → 4 iha painel.

## Fluxu servisu iha painel

| Tab | Saida mak halo |
|---|---|
| **📥 1. Pedidu Dadus** | Pedidu manuál no pedidu automátiku (padraun kada 30 minutu). Kada pedidu foti notísia to'o ficheiru CSV 200 MB. |
| **🧪 2. Preprosesamentu** | Importa, hamoos, case folding, tokenizasaun, hasai stopword, stemming, kategoria referénsia. Rai dadus bersih. |
| **🧠 3. Treinu Modelu** | Vetorizasaun TF-IDF, klaster K-Means, identifikasaun tópiku, avaliasaun, vizualizasaun, tópiku populár, lista liafuan. |
| **📊 4. Analiza Notísia** | Tópiku, tendénsia, kalohan liafuan, notísia, galeria, istória (husi modelu ativu). |

### 1. Pedidu dadus

- Pedidu ida foti notísia husi orijen aktivu (WordPress REST API) to'o CSV to'o meta (padraun **200 MB**).
- Kada pedidu rai iha folder rasik `data/batches/batch_AAAAMMDD_HHMMSS/` ho `notisia.csv`, `notisia.json`,
  `notisia.db` (SQLite) no `meta.json`. Lote **la hamutuk** ho lote seluk, atu bele treina kada lote.
- Notísia ne'ebé rai ona la repete iha lote tuir mai (indeks `data/batches/index.db`). Pedidu foun foti uluk
  notísia foun, depois kontinua ba notísia tuan.
- **Pedidu automátiku**: ativa iha tab 1, intervalu padraun 30 minutu, hahú depois pedidu anterior remata. Hala'o
  iha background, maski tab browser taka, kuandu servisu painel sei moris. Pedidu manuál sempre bele hala'o.
- Se orijen la iha notísia foun tan, lote sei ki'ik liu 200 MB no marka "Meta to'o: Lae". Se la iha notísia foun,
  la kria lote.
- Lote ida uza besik 3 dala tamañu CSV iha disku (CSV + JSON + SQLite). Hamoos lote tuan se disku nakonu.

### 2. Preprosesamentu

Hili lote husi pedidu (CSV, JSON ka SQLite) ka karga ficheiru rasik. Etapa:
importa → hamoos (HTML, entidade, URL) → case folding → tokenizasaun → deteksaun lian no hasai stopword →
stemming (Tetun, Indonézia, Inglés) → kategoria referénsia → rai. Prosesa ba lote 5.000 notísia atu memória la nakonu.
Rezultadu rai iha `data/cleaned/bersih_<naran>/` ho `notisia_bersih.csv`, `.json`, `.db` no `meta.json`.

### 3. Treinu modelu

Hili dadus bersih ka karga ficheiru bersih (tenke iha kolona `text_final`). Treinu hala'o ho etapa:
TF-IDF (fit iha dadus treinu) → K-Means (k otomátiku ho silhouette ka k hili rasik) → tópiku (liafuan-xave, label) →
avaliasaun ho dadus teste → rai modelu iha `data/models/versions/<versaun>/`. Modelu antigu kontinua ativu se treinu falla.

**Akurasia, presizaun, recall no F1-score**: K-Means la iha label, entaun kada klaster simu kategoria dominante
husi dadus treinu, depois kompara ho kategoria referénsia iha dadus teste. Kategoria referénsia mai husi klasifikadór
liafuan-xave (Edukasaun, Politika, Saúde, Ekonomia, Jerál), la'ós label umanu. Métrika sukat konkordánsia ho
referénsia ne'e. Painel mós hatudu akurasia baze (prediz kategoria dominante hotu) atu bele kompara.

Rezultadu: métrika, matriz konfuzaun, silhouette kada k, tópiku, **tópiku populár** (kompara janela ikus ho janela
molok), tendénsia tempu, **lista liafuan** (global no kada tópiku, bele foti CSV).

## Estrutura folder

```
data/batches/    lote pedidu (CSV, JSON, SQLite), state.json, index.db, auto_state.json
data/cleaned/    dadus bersih (kada folder: CSV, JSON, SQLite, meta.json)
data/models/     versaun modelu (TF-IDF, K-Means, dadus ho klaster, meta.json)
data/uploads/    ficheiru ne'ebé karga husi painel
src/collection/  koletór lote 200 MB
src/storage/     hakerek CSV + JSON + SQLite
src/jobs/        servisu background no jadwal automátiku
src/preprocessing/staged_pipeline.py   preprosesamentu ho etapa
src/training/    treinu, avaliasaun, análize tópiku populár
src/dashboard/   painel Streamlit (workflow_views.py = tab 1 to'o 3)
config/settings.yaml   konfigurasaun (seksaun `batch` no `training`)
```

## Konfigurasaun

`config/settings.yaml`: seksaun `batch` (meta MB, intervalu automátiku, ativa/dezativa), `training`
(parámetru TF-IDF no K-Means), `sources` (orijen). Limite karga ficheiru iha `.streamlit/config.toml`.

## Nota importante

- Pedidu foti dadus husi WordPress REST API orijen sira. Se API taka, koletór uza RSS (notísia ki'ik).
- Pedidu 200 MB bele demora minutu barak (jeralmente 100 notísia kada pájina, ho intervalu 1 segundu).
- Pedidu automátiku presiza servisu painel sei moris; loke painel dala ida hodi hahú servisu.
- Servisu terminál antigu (`python main.py`) sei bele uza, maibé la presiza tan.

## Teste

```bash
python -m pytest tests -v
```
Teste `tests/integration/test_dashboard_app.py` uza `streamlit.testing`.

## Instalasaun manuál

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run src/dashboard/trend_dashboard.py
```
