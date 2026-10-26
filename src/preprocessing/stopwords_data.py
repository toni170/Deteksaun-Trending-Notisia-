"""
Daftar stopwords untuk 3 bahasa yang umum dipakai di berita Tatoli:
Tetun (tet), Bahasa Indonesia (id), dan Inggris (en).

Dikurasi manual (bukan lewat download eksternal) supaya sistem tetap bisa
berjalan sepenuhnya offline setelah RSS diambil.
"""

STOPWORDS_ID = {
    "yang", "untuk", "pada", "ke", "para", "namun", "menurut", "antara", "dia",
    "dua", "ia", "seperti", "jika", "sehingga", "kembali", "dan", "tidak",
    "ini", "karena", "kepada", "oleh", "saat", "harus", "sementara", "setelah",
    "belum", "kami", "sekitar", "bagi", "serta", "di", "dari", "telah", "sebagai",
    "masih", "hal", "ketika", "adalah", "itu", "dalam", "bisa", "akan", "sudah",
    "dengan", "atau", "juga", "yaitu", "saja", "lagi", "kita", "kamu", "aku",
    "anda", "mereka", "kalau", "agar", "supaya", "bagaimana", "apakah", "apa",
    "siapa", "kenapa", "mengapa", "dimana", "kapan", "semua", "banyak",
    "sedikit", "lebih", "kurang", "sangat", "sekali", "satu", "dua", "tiga",
    "kali", "waktu", "orang", "tahun", "hari", "bulan", "saya", "engkau", "kau",
    "beliau", "nya", "mu", "ku", "berita", "wartawan", "laporan", "pukul",
}

STOPWORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're",
    "you've", "you'll", "you'd", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "she's", "her", "hers", "herself",
    "it", "it's", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "that'll",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "s", "t", "can", "will", "just", "don", "don't", "should",
    "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain",
    "said", "news", "report", "reported",
}

# Kurasi manual stopwords Tetun (bahasa nasional Timor-Leste)
STOPWORDS_TET = {
    "ne'e", "nee", "ne", "sira", "ita", "ha'u", "hau", "ó", "o", "nia", "nian",
    "atu", "ba", "husi", "la", "lae", "ho", "iha", "tuir", "hodi", "maka",
    "duni", "tan", "mós", "mos", "karik", "wainhira", "bainhira", "se",
    "nu'udar", "nudar", "hanesan", "katak", "los", "no", "maibe", "maibé",
    "husik", "kona", "ba'i", "bele", "labele", "presiza", "tenki", "sei",
    "seidauk", "ona", "hela", "hotu", "tebes", "loos", "diak", "aat", "boot",
    "kiik", "ida", "rua", "tolu", "haat", "lima", "loron", "fulan", "tinan",
    "ema", "povu", "governu", "nasaun", "amu", "aman", "inan", "alin", "maun",
    "bin", "oan", "sé", "buat", "razaun", "tanba", "entaun", "deit", "so",
    "koalia", "hatete", "dehan", "hanoin", "sente", "jornalista", "notisia",
    "kona ba",
}
