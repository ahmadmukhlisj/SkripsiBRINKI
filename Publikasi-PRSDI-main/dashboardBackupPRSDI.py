import streamlit as st
import plotly.express as px
import pandas as pd
from pyvis.network import Network
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import io
from IPython.core.display import HTML
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

st.set_page_config(layout="wide")
st.title('Dashboard Capaian KI PRSDI')
# Refresh page every 60 seconds
st_autorefresh(interval=60 * 1000)

@st.cache_data
def get_data(sheets):
    # Tentukan scope untuk mengakses Google Sheets dan Google Drive
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    

    # Autentikasi menggunakan secrets dari Streamlit Cloud
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=scope
    )
    
    client = gspread.authorize(credentials)

    spreadsheet = client.open("Form Capaian PRSDI 2024")
    sheet = spreadsheet.get_worksheet(sheets)
    data = sheet.get_all_values()
    df_KI = pd.DataFrame(data[1:], columns=data[0])
    return df_KI

# Sheets Kekayaan Intelektual
# df_KI = pd.read_excel('Form Capaian PRSDI.xlsx', sheet_name='KI')
# dfl_KI = get_data()
df_KI = get_data(1)
df_sivitas = pd.read_excel('Sivitas_PRSDI.xlsx')

#Data Preprocessing
def hapus_baris_kosong(df, kolom):
    # Ganti string kosong atau whitespace dengan NaN
    df[kolom] = df[kolom].replace(r'^\s*$', None, regex=True)
    
    # Hapus baris yang memiliki nilai NaN pada kolom tertentu
    df_bersih = df.dropna(subset=[kolom])
    
    return df_bersih

def formatting_data(df_ki):
    df_KI.columns = df_KI.iloc[3]
    df_ki = df_KI.iloc[4:].reset_index(drop=True)
    df_ki.columns.name = None
    # df_ki['Periode Input'] = pd.to_datetime(df_ki['Periode Input'], format='%B %d, %Y', errors='coerce')
    return df_ki

def rename_columns(df_ki):
    new_columns = ['Inventor ' + str(i) for i in range(1, 9)]
    df_ki.columns = list(df_ki.columns[:6]) + new_columns + list(df_ki.columns[14:])
    return df_ki

def drop_data(df_ki):
    df_ki = df_ki.drop(df_ki.columns[27:31], axis=1)
    # Drop kolom pada indeks 25
    # df_ki = df_ki.iloc[:, :-1]
    
    # Hapus baris yang mengandung nilai null pada kolom 'Judul'
    df_ki = df_ki.dropna(subset=['JUDUL'])
    df_ki = df_ki.dropna(how='all')  # Hapus baris dengan semua nilai null
    return df_ki
    
# Kamus bulan
bulan_dict = {
    'Januari': 'January', 'Februari': 'February', 'Maret': 'March',
    'April': 'April', 'Mei': 'May', 'Juni': 'June',
    'Juli': 'July', 'Agustus': 'August', 'September': 'September',
    'Oktober': 'October', 'November': 'November', 'Desember': 'December'
}

# Fungsi untuk mengonversi nama bulan
def convert_month(date_str):
    for id_ind, id_eng in bulan_dict.items():
        if id_ind in date_str:
            return date_str.replace(id_ind, id_eng)
    return date_str
    
# Fungsi untuk mengatur pewarnaan baris
def highlight_rows(df_KI):
    if df_KI['JENIS'] == 'Hak Cipta':
        return ['background-color: #83c9ff']*len(df_KI)
    elif df_KI['JENIS'] == 'Paten':
        return ['background-color: #1E90FF']*len(df_KI)
    else:
        return ['']*len(df_KI)

# Menggunakan cache untuk data preprocessing
def preprocessing(df_KI):
    df_ki = formatting_data(df_KI)
    df_ki = rename_columns(df_ki)
    df_ki = drop_data(df_ki)
#   df_ki = hapus_baris_kosong(df_ki, "Periode Input")
    df_ki = hapus_baris_kosong(df_ki, "JUDUL")
    df_ki = hapus_baris_kosong(df_ki, "JENIS")
    df_ki['Periode Input'] = df_ki['Periode Input'].apply(convert_month)
    df_ki['Periode Input'] = pd.to_datetime(df_ki['Periode Input'], format='%B %d, %Y', errors='coerce')
    df_ki['TANGGAL PENDAFTARAN'] = df_ki['TANGGAL PENDAFTARAN'].replace(r'^\s*$', None, regex=True)
    df_ki['TANGGAL PENDAFTARAN'] = df_ki['TANGGAL PENDAFTARAN'].fillna('Belum Terdaftar')
    df_ki['TANGGAL SERTIFIKAT'] = df_ki['TANGGAL SERTIFIKAT'].replace(r'^\s*$', None, regex=True)
    df_ki['TANGGAL SERTIFIKAT'] = df_ki['TANGGAL SERTIFIKAT'].fillna('Belum Tersertifikasi')
    return df_ki

#Preprocessing Data
df_ki = preprocessing(df_KI)

# Fungsi pie_chart yang sudah Anda buat
def pie_chart(df_ki):
    # Group by 'JENIS' and 'STATUS', then count the occurrences
    jenis_status_counts = df_ki.groupby(['JENIS', 'STATUS'])['JUDUL'].count().reset_index(name='Count')

    # Create a list of unique statuses
    statuses = jenis_status_counts['STATUS'].unique()

    # Create a figure with the hole for the donut chart
    fig = px.pie(hole=0.3)  # Set the hole property here

    # Add traces for each status
    for status in statuses:
        filtered_data = jenis_status_counts[jenis_status_counts['STATUS'] == status]
        fig.add_trace(px.pie(filtered_data, values='Count', names='JENIS', hole=0.3).data[0])

    # Add trace for all data
    all_data = df_ki.groupby('JENIS')['JUDUL'].count().reset_index(name='Count')
    fig.add_trace(px.pie(all_data, values='Count', names='JENIS', hole=0.3).data[0])
    fig.update_traces(textinfo='label+value+percent', textposition='inside')

    # Update layout to include dropdown (remove 'hole' from here)
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "All",
                        "method": "update",
                        "args": [{"visible": [True] + [False]*len(statuses)},
                                {"title": "Jumlah Jurnal/Prosiding per Jenis - All"}]
                    }
                ] + [
                    {
                        "label": status,
                        "method": "update",
                        "args": [{"visible": [False] + [status == s for s in statuses]},
                                {"title": f"Jumlah Jurnal/Prosiding per Jenis - {status}"}]
                    }
                    for status in statuses
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.95,  # Position x
                "y": 0.98,  # Position y
                "xanchor": "right",
                "yanchor": "top"
            }
        ]
    )

    return fig

def bar_chart(df_ki):
    # Group by 'JENIS' and count the occurrences
    jenis_counts = df_ki.groupby('JENIS')['JUDUL'].count().reset_index(name='Count')

    # Create the bar chart using Plotly Express with color and text labels
    fig = px.bar(jenis_counts, x='JENIS', y='Count',
                 labels={'JENIS': 'Jenis Kekayaan Intelektual', 'Count': 'Jumlah'},
                 color='JENIS', text='Count')  # Menambahkan parameter text untuk label
                

    # Atur posisi teks label di luar batang
    fig.update_traces(textposition='outside')

    return fig

def submission_trend(df_ki):
    # Buat layout dengan kolom untuk memposisikan filter di pojok kanan atas
    col1, col2 = st.columns([6, 1])

    with col2:
        # Tambahkan dropdown filter dengan ukuran lebih kecil di kolom kanan
        filter_option = st.selectbox(
            'Filter Waktu',
            ('Per Hari', 'Per Minggu', 'Per Bulan')  # Menghilangkan label default
        )

    # Menghitung jumlah jurnal yang diinput per Tanggal
    df_count = df_ki.groupby(df_ki['Periode Input'].dt.date).size().reset_index(name='Jumlah Jurnal')

    # Mengonversi kembali kolom 'Periode Input' menjadi datetime
    df_count['Periode Input'] = pd.to_datetime(df_count['Periode Input'])

    # Mengatur 'Periode Input' sebagai index untuk melakukan resampling
    df_count.set_index('Periode Input', inplace=True)

    # Filter berdasarkan pilihan
    if filter_option == 'Per Hari':
        df_count = df_count.resample('D').sum().reset_index()
    elif filter_option == 'Per Minggu':
        df_count = df_count.resample('W').sum().reset_index()
    elif filter_option == 'Per Bulan':
        df_count = df_count.resample('M').sum().reset_index()

    # Filter untuk label non-0
    df_count['Label'] = df_count['Jumlah Jurnal'].apply(lambda x: x if x != 0 else None)

    # Membuat line chart menggunakan Plotly dengan label hanya pada titik yang bukan 0
    fig = px.line(
        df_count, 
        x='Periode Input', 
        y='Jumlah Jurnal', 
        title=f'Trend Jumlah Kekayaan Intelektual ({filter_option})',
        labels={
            'Periode Input': 'Periode Waktu',  # Label untuk sumbu x
            'Jumlah Jurnal': 'Jumlah Jurnal'   # Label untuk sumbu y
        },
        text='Label'  # Menampilkan hanya label yang bukan 0
    )

    # Mengatur agar hanya menampilkan garis dan label, tanpa bulatan
    fig.update_traces(textposition='top center', mode='lines+text')

    # Tampilkan chart di Streamlit
    st.plotly_chart(fig)

def bar_chart_publikasi(df_ki, df_sivitas):
    df_ki_inventor = df_ki[['JUDUL', 'JENIS', 'Inventor 1',
        'Inventor 2', 'Inventor 3', 'Inventor 4', 'Inventor 5',
        'Inventor 6', 'Inventor 7', 'Inventor 8']]

    global df_ki_writer
    df_ki_writer = df_ki_inventor

    # Create an empty dictionary to store the counts
    author_counts = {}

    # Iterate through each row in the DataFrame
    for index, row in df_ki_inventor.iterrows():
        for i in range(1, 9):
            author_name = row['Inventor ' + str(i)]
            if pd.notna(author_name):  # Check if the author name is not NaN
                jenis = row['JENIS']
                if author_name not in author_counts:
                    author_counts[author_name] = {'Hak Cipta': 0, 'Paten': 0}
                if jenis == 'Hak Cipta':
                    author_counts[author_name]['Hak Cipta'] += 1
                elif jenis == 'Paten':
                    author_counts[author_name]['Paten'] += 1

    # Convert the dictionary to a DataFrame
    df_ki_authors = pd.DataFrame.from_dict(author_counts, orient='index')
    df_ki_authors.index.name = 'Nama'
    df_ki_authors = df_ki_authors.reset_index()
    df_ki_authors = hapus_baris_kosong(df_ki_authors, "Nama")
    
    global df_ki_auth
    df_ki_auth = df_ki_authors

    # Ubah format data
    df_melted = pd.melt(df_ki_authors, id_vars=['Nama'], value_vars=['Hak Cipta', 'Paten'],
                        var_name='Publikasi', value_name='Jumlah')

    # Menghapus baris dengan Jumlah 0
    df_melted = df_melted[df_melted['Jumlah'] != 0].reset_index(drop=True)

    # Merge data dengan df_sivitas
    df_melted = pd.merge(df_melted, df_sivitas[['Nama', 'Kelompok Riset']], on='Nama', how='left')
    df_melted['Kelompok Riset'] = df_melted['Kelompok Riset'].fillna('Unknown')
    df_melted = df_melted.sort_values(by='Jumlah')

    # Filter Kelompok Riset
    col1, col2 = st.columns([3, 1])

    with col2:
        # Tambahkan dropdown filter di kolom kanan
        kelompok_riset_list = df_melted['Kelompok Riset'].unique().tolist()
        selected_kelompok_riset = st.selectbox('Pilih Kelompok Riset:', ['Semua'] + kelompok_riset_list)

    if selected_kelompok_riset != 'Semua':
        df_melted = df_melted[df_melted['Kelompok Riset'] == selected_kelompok_riset]

    # Plotting the filtered data with label outside the bars
    fig = px.bar(df_melted, x="Jumlah", y="Nama", color="Publikasi", title="Jumlah Kekayaan Intelektual per Inventor", text='Jumlah')

    # Set label text position to outside the bars
    fig.update_traces(textposition='outside', textangle=0)

    return fig

def bar_chart_terdaftar(df_ki, df_sivitas):
    # Ambil kolom yang relevan
    df_ki_inventor = df_ki[['JUDUL', 'JENIS', 'TANGGAL PENDAFTARAN']]

    global df_ki_writer
    df_ki_writer = df_ki_inventor

    # Create a dictionary to count publications by registration date
    date_counts = {}

    # Iterate through each row in the DataFrame
    for index, row in df_ki_inventor.iterrows():
        tanggal = row['TANGGAL PENDAFTARAN']
        jenis = row['JENIS']
        
        if pd.notna(tanggal):  # Check if the date is not NaN
            if tanggal not in date_counts:
                date_counts[tanggal] = {'Hak Cipta': 0, 'Paten': 0}
            if jenis == 'Hak Cipta':
                date_counts[tanggal]['Hak Cipta'] += 1
            elif jenis == 'Paten':
                date_counts[tanggal]['Paten'] += 1

    # Convert the dictionary to a DataFrame
    df_ki_dates = pd.DataFrame.from_dict(date_counts, orient='index')
    df_ki_dates.index.name = 'Tanggal Pendaftaran'
    df_ki_dates = df_ki_dates.reset_index()

    # Ubah format data untuk plotting
    df_melted = pd.melt(df_ki_dates, id_vars=['Tanggal Pendaftaran'], value_vars=['Hak Cipta', 'Paten'],
                        var_name='Publikasi', value_name='Jumlah')

    # Menghapus baris dengan Jumlah 0
    df_melted = df_melted[df_melted['Jumlah'] != 0].reset_index(drop=True)

    # Merge data dengan df_sivitas jika diperlukan (dalam konteks ini tidak digunakan)
    # df_melted = pd.merge(df_melted, df_sivitas[['Nama', 'Kelompok Riset']], on='Nama', how='left')

    # Plotting the filtered data with label outside the bars
    fig = px.bar(df_melted, x="Jumlah", y="Tanggal Pendaftaran", color="Publikasi", 
                 title="Jumlah Kekayaan Intelektual Yang Sudah Terdaftar", text='Jumlah')

    # Set label text position to outside the bars
    fig.update_traces(textposition='outside', textangle=0)

    return fig

def bar_chart_tersertifikat(df_ki, df_sivitas):
    # Ambil kolom yang relevan
    df_ki_inventor = df_ki[['JUDUL', 'JENIS', 'TANGGAL SERTIFIKAT']]

    global df_ki_writer
    df_ki_writer = df_ki_inventor

    # Create a dictionary to count publications by certificate date
    date_counts = {}

    # Iterate through each row in the DataFrame
    for index, row in df_ki_inventor.iterrows():
        tanggal = row['TANGGAL SERTIFIKAT']
        jenis = row['JENIS']
        
        if pd.notna(tanggal):  # Check if the date is not NaN
            if tanggal not in date_counts:
                date_counts[tanggal] = {'Hak Cipta': 0, 'Paten': 0}
            if jenis == 'Hak Cipta':
                date_counts[tanggal]['Hak Cipta'] += 1
            elif jenis == 'Paten':
                date_counts[tanggal]['Paten'] += 1

    # Convert the dictionary to a DataFrame
    df_ki_dates = pd.DataFrame.from_dict(date_counts, orient='index')
    df_ki_dates.index.name = 'Tanggal Sertifikat'
    df_ki_dates = df_ki_dates.reset_index()

    # Ubah format data untuk plotting
    df_melted = pd.melt(df_ki_dates, id_vars=['Tanggal Sertifikat'], value_vars=['Hak Cipta', 'Paten'],
                        var_name='Publikasi', value_name='Jumlah')

    # Menghapus baris dengan Jumlah 0
    df_melted = df_melted[df_melted['Jumlah'] != 0].reset_index(drop=True)

    # Merge data dengan df_sivitas jika diperlukan (dalam konteks ini tidak digunakan)
    # df_melted = pd.merge(df_melted, df_sivitas[['Nama', 'Kelompok Riset']], on='Nama', how='left')

    # Plotting the filtered data with label outside the bars
    fig = px.bar(df_melted, x="Jumlah", y="Tanggal Sertifikat", color="Publikasi", 
                 title="Jumlah Kekayaan Intelektual Yang Sudah Tersertifikasi", text='Jumlah')

    # Set label text position to outside the bars
    fig.update_traces(textposition='outside', textangle=0)

    return fig

def plot_barchart(df, col):
    # Ganti string kosong atau whitespace dengan NaN
    df[col] = df[col].replace(r'^\s*$', None, regex=True)
    # Ganti nilai null dengan 'Unknown'
    df[col] = df[col].fillna('Unknown')

    # Hitung jumlah masing-masing kelompok
    count_df = df[col].value_counts().reset_index()
    count_df.columns = [col, 'Jumlah']
    # Urutkan dari besar ke kecil
    count_df = count_df.sort_values(by='Jumlah', ascending=True)

    # Buat bar chart dengan Plotly
    fig = px.bar(count_df, x='Jumlah', y=col, color='Jumlah',
                 color_continuous_scale='Blues', text='Jumlah')

    # Atur posisi teks label di luar batang
    fig.update_traces(textposition='outside')

    return fig

def plot_piechart(df):
    # Hitung jumlah masing-masing kategori reputasi
    count_df = df['NO PENDAFTARAN'].value_counts().reset_index()
    count_df.columns = ['TANGGAL SERTIFIKAT', 'Jumlah']
    count_df = df['TANGGAL PENDAFTARAN'].value_counts().reset_index()
    count_df.columns = ['TANGGAL PENDAFTARAN', 'Jumlah']
    
    terisi = df_ki['TANGGAL PENDAFTARAN'].notnull().sum()
    kosong = df_ki['TANGGAL PENDAFTARAN'].isnull().sum()
    count_df = pd.DataFrame({'TANGGAL PENDAFTARAN': ['Terisi', 'Kosong'], 'Jumlah': [terisi, kosong]})
    
    # Buat Bar chart dengan Plotly
    fig = px.pie(count_df, names='TANGGAL SERTIFIKAT', values='Jumlah', hole=0.3)
    fig = px.pie(count_df, names='TANGGAL PENDAFTARAN', values='Jumlah', hole=0.3)
    return fig

def network_graph(df_ki_authors, df_ki_inventor):
    graph_data = []
    for index, row in df_ki_authors.iterrows():
        for _ in range(row['Hak Cipta']):
            graph_data.append({'Nama': row['Nama'], 'Kekayaan Intelektual': 'Hak Cipta'})
        for _ in range(row['Paten']):
            graph_data.append({'Nama': row['Nama'], 'Kekayaan Intelektual': 'Paten'})

    df_authors = pd.DataFrame(graph_data)

    # Mengubah dataframe
    results = []

    for index, row in df_ki_inventor.iterrows():
        for penulis in row[2:]:
            if pd.notna(penulis):
                results.append({"Source": penulis, "Target": row["JUDUL"], "Jenis": row["JENIS"]})

    new_df = pd.DataFrame(results)

    # Mengatasi nilai NaN pada kolom 'Jenis'
    new_df['Jenis'] = new_df['Jenis'].fillna('Unknown')
    new_df = hapus_baris_kosong(new_df, "Source")

    # Membuat network graph
    net = Network(height='750px', width='100%', notebook=False, cdn_resources='in_line', filter_menu=True, select_menu=True)

    # Menambahkan node dari kolom Source dengan warna pink
    for source in new_df['Source'].unique():
        net.add_node(source, color='pink')

    # Menambahkan node dari kolom Target dengan warna berdasarkan Jenis
    color_map = {
        'Hak Cipta': 'orange',
        'Paten': 'yellow',
        'Unknown': 'gray'  # Warna default untuk nilai yang tidak dikenal
    }

    for idx, row in new_df.iterrows():
        net.add_node(row['Target'], color=color_map[row['Jenis']])
        net.add_edge(row['Source'], row['Target'])

    # Menghitung derajat untuk setiap node
    degree_dict = {}
    for node in net.nodes:
        node_id = node['id']
        degree_dict[node_id] = 0

    for edge in net.edges:
        degree_dict[edge['from']] += 1
        degree_dict[edge['to']] += 1

    # Menetapkan ukuran node dan hover text berdasarkan derajat
    max_degree = max(degree_dict.values(), default=1)  # untuk menghindari pembagian dengan 0

    for node in net.nodes:
        node_id = node['id']
        degree = degree_dict.get(node_id, 0)
        # Menetapkan ukuran node berdasarkan derajat, menyesuaikan skalanya jika diperlukan
        node['size'] = (degree / max_degree) * 30 + 10  # ukuran minimum 10, skala maksimum 30
        # Menambahkan hover text dengan ID node dan jumlah degree
        node['title'] = f"{node_id} dengan jumlah degree: {degree}"

    # Menampilkan tombol kontrol fisika
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)

    # Menyimpan network graph ke file HTML
    # Buat HTML string
    net_html = net.generate_html()

    # Simpan dengan encoding utf-8
    with io.open('network_graph_w.html', 'w', encoding='utf-8') as f:
        f.write(net_html)
#   net.save_graph('network_graph_w.html')

#def generate_wordcloud(dataframe, column_name):
    # Gabungkan semua teks di kolom menjadi satu string
    #text = " ".join(title for title in dataframe[column_name])
    
    # Define the stopwords set including the word to exclude
    #stopwords = set(WordCloud().stopwords)
    #stopwords.add('Using')
    #stopwords.add('Analysis')
    #stopwords.add('Data')
    #stopwords.add('Model')
    #stopwords.add('Based')
    #stopwords.add('Indonesia')
    #stopwords.add('Analytics')

    # Membuat WordCloud
    #wordcloud = WordCloud(width=1920, height=1080, background_color='white', colormap='winter_r', stopwords=stopwords).generate(text)

    # Tampilkan WordCloud menggunakan Matplotlib tanpa outline
    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax.imshow(wordcloud, interpolation='bilinear')
    #ax.axis('off')
    
    # Tampilkan di Streamlit
    #st.pyplot(fig)

def generate_wordcloud(dataframe, column_name):
    # Gabungkan semua teks di kolom menjadi satu string
    text = " ".join(title for title in dataframe[column_name])

    # Define the stopwords set including the word to exclude
    stopwords = set(WordCloud().stopwords)
    stopwords.add('Using')
    stopwords.add('Analysis')
    stopwords.add('Data')
    stopwords.add('Model')
    stopwords.add('Based')
    stopwords.add('Indonesia')
    stopwords.add('Analytics')
    stopwords.add('dan')
    stopwords.add('dengan')
    stopwords.add('OK')
    stopwords.add('dalam')
    stopwords.add('untuk')
    stopwords.add('pada')
    stopwords.add('ke')
    stopwords.add('di')
    stopwords.add('selanjutnya')

    # Membuat WordCloud
    wordcloud = WordCloud(width=1920, height=1080, background_color='white', colormap='winter_r', stopwords=stopwords).generate(text)

    # Tampilkan WordCloud menggunakan Matplotlib tanpa outline
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    # Tampilkan di Streamlit
    st.pyplot(fig)

    # Implementasi TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(dataframe['JUDUL'])
    feature_names = tfidf_vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    df_tfidf = pd.DataFrame(denselist, columns=feature_names)

    # Menampilkan kata-kata penting berdasarkan TF-IDF
    important_words = df_tfidf.sum().sort_values(ascending=False).head(10)
    df_important_words = important_words.reset_index()
    df_important_words.columns = ['Kata', 'Bobot']

   # st.write("Kata-kata penting berdasarkan TF-IDF:")
   # st.write(important_words)

    # Implementasi LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tfidf_matrix)

    # Menampilkan topik yang dihasilkan oleh LDA
    #for index, topic in enumerate(lda.components_):
    #    top_features_ind = topic.argsort()[-10:][::-1]
    #    top_features = [feature_names[i] for i in top_features_ind]
    #    st.write(f"Topik {index + 1}: {', '.join(top_features)}")
    topics = []
    lingkup = [
        "Informasi Geografis",
        "Analisis",
        "Pengolahan",
        "Teknologi",
        "Proses"
    ]    
    for index, topic in enumerate(lda.components_):
        top_features_ind = topic.argsort()[-10:][::-1]  # Indentasi yang benar
        top_features = [feature_names[i] for i in top_features_ind]  # Indentasi yang benar
        topics.append([f"Topik {index + 1}", lingkup[index], ', '.join(top_features)])  # Menggunakan lingkup yang sesuai

    # Membuat DataFrame untuk topik
    df_topics = pd.DataFrame(topics, columns=['Topik', 'Lingkup', 'Kata-Kata'])
    #Create Two Columns
    #col1, col2 = st.columns(2)
    #with col1:
        # Menampilkan tabel TF-IDF
        #st.write("Kata-kata penting berdasarkan TF-IDF:")
        #st.write(df_important_words)
    #with col2: 
    # Menampilkan tabel
    st.write("Topik yang dihasilkan oleh LDA:")
    st.write(df_topics)

def network(df_ki):
    df_graph = df_ki[['JUDUL', 'KELOMPOK RISET']]
    
    # Buat network
    net = Network(height='750px', width='100%', notebook=False, cdn_resources='in_line', filter_menu=True)

    # Hitung degree untuk setiap node
    degree_count = {}

    # Loop untuk menghitung degree
    for index, row in df_graph.iterrows():
        degree_count[row['JUDUL']] = degree_count.get(row['JUDUL'], 0) + 1
        degree_count[row['KELOMPOK RISET']] = degree_count.get(row['KELOMPOK RISET'], 0) + 1

    # Dapatkan degree maksimum dan minimum
    max_degree = max(degree_count.values())
    min_degree = min(degree_count.values())

    # Fungsi untuk mendapatkan warna gradien berdasarkan degree
    def get_color_gradient(degree):
        normalized_degree = (degree - min_degree) / (max_degree - min_degree)
        # Gradien warna dari biru (degree terendah) ke merah (degree tertinggi)
        return plt.cm.coolwarm(normalized_degree)

    # Tambahkan node dan edge ke dalam network
    for index, row in df_graph.iterrows():
        # Hover text untuk node JUDUL
        hover_text_judul = f"{row['JUDUL']} dengan jumlah degree: {degree_count[row['JUDUL']]}"

        # Hover text untuk node KELOMPOK RISET
        hover_text_kelompok = f"{row['KELOMPOK RISET']} dengan jumlah degree: {degree_count[row['KELOMPOK RISET']]}"

        # Tambahkan node source (JUDUL) dengan warna tetap
        net.add_node(row['JUDUL'], title=hover_text_judul, color='pink', size=degree_count[row['JUDUL']]*15)  # Source node
        
        # Dapatkan warna gradien untuk node target berdasarkan degree
        color_kelompok = get_color_gradient(degree_count[row['KELOMPOK RISET']])
        hex_color_kelompok = f"#{int(color_kelompok[0]*255):02x}{int(color_kelompok[1]*255):02x}{int(color_kelompok[2]*255):02x}"

        # Tambahkan node target (KELOMPOK RISET) dengan warna gradien
        net.add_node(row['KELOMPOK RISET'], title=hover_text_kelompok, color=hex_color_kelompok, size=degree_count[row['KELOMPOK RISET']]*5)  # Target node
        
        # Buat edge antara source dan target
        net.add_edge(row['JUDUL'], row['KELOMPOK RISET'])

    # Terapkan algoritma force atlas untuk tata letak yang lebih baik
    net.force_atlas_2based(gravity=-50, central_gravity=0.005, spring_length=300, spring_strength=0.08, damping=0.4, overlap=0)
    net_html = net.generate_html()

    # Simpan ke file HTML dengan encoding utf-8
    with io.open('network.html', 'w', encoding='utf-8') as f:
        f.write(net_html)


# Streamlit App
def main():
    """### **Data Kekayaan Intelektual**"""
    st.dataframe(df_ki)

    # Menampilkan Pie Chart dan Bar Chart
    
    fig_pie = pie_chart(df_ki)
    fig_bar = bar_chart(df_ki)
    fig_bar1 = plot_barchart(df_ki, 'KELOMPOK RISET')
    fig_bar2 = plot_barchart(df_ki, 'TANGGAL SERTIFIKAT')
    fig_bar4 = plot_barchart(df_ki, 'TANGGAL PENDAFTARAN')
    fig_bar3 = plot_barchart(df_ki, 'STATUS')

    # Create two columns
    col1, col2 = st.columns(2)

    # Display pie chart in the first column
    with col1:
        st.subheader("Pie Chart Sebaran Jenis Kekayaan Intelektual")
        st.plotly_chart(fig_pie)
    # Display bar chart in the second column
    with col2:
        st.subheader("Bar Chart Perbandingan Jenis Kekayaan Intelektual")
        st.plotly_chart(fig_bar)

    #Status KI
    st.subheader("Status Kekayaan Intelektual")
    st.plotly_chart(fig_bar3)

    # Create three columns
    col1, col2 = st.columns(2)

    # Display pie chart in the first column
    with col2:
        st.subheader("KI yang Tersertifikasi")
       # st.plotly_chart(fig_pie1)
        st.plotly_chart(fig_bar2)

    # Display bar chart in the second column
    with col1:
       st.subheader("KI yang Terdaftar")
       st.plotly_chart(fig_bar4)


# Menampilkan Bar Chart Tersertifikasi
   # st.subheader("Bar Chart KI Terdaftar")
    fig_daftar = bar_chart_terdaftar(df_ki, df_sivitas)
    st.plotly_chart(fig_daftar)

# Menampilkan Bar Chart Tersertifikasi
  #  st.subheader("Bar Chart KI Tersertifikasi")
    fig_sertifikasi = bar_chart_tersertifikat(df_ki, df_sivitas)
    st.plotly_chart(fig_sertifikasi)
    
    
      # Membuat 2 Kolom Baru
    col1, col2 = st.columns(2)

    # Display pie chart in the first column
    with col1:
        st.subheader("Judul KI yang Terdaftar")
        df_ki['NO PENDAFTARAN'] = df_ki['NO PENDAFTARAN'].replace(r'^\s*$', None, regex=True)
        df_filtered = df_ki[['JUDUL', 'JENIS', 'NO PENDAFTARAN']]
        df_filtered = hapus_baris_kosong(df_filtered, 'NO PENDAFTARAN')
        df_styled = df_filtered.style.apply(highlight_rows, axis=1)
        st.dataframe(df_styled, use_container_width=True)
    with col2:
        st.subheader("Judul KI yang Tersertifikasi")
        df_ki['NO SERTIFIKAT'] = df_ki['NO SERTIFIKAT'].replace(r'^\s*$', None, regex=True)
        df_filtered = df_ki[['JUDUL', 'JENIS', 'NO SERTIFIKAT']]
        df_filtered = hapus_baris_kosong(df_filtered, 'NO SERTIFIKAT')
        df_styled = df_filtered.style.apply(highlight_rows, axis=1)
        st.dataframe(df_styled, use_container_width=True)
        
    #Bar Chart Kelompok Riset
    st.subheader("Bar Chart Kelompok Riset")
    st.plotly_chart(fig_bar1)

    # Menampilkan Bar Chart Publikasi
    st.subheader("Bar Chart Kekayaan Intelektual per Inventor")
    fig_publikasi = bar_chart_publikasi(df_ki, df_sivitas)
    st.plotly_chart(fig_publikasi)

    # Menampilkan Trend Input Data
    st.subheader("Trend Input Data Kekayaan Intelektual")
    submission_trend(df_ki)

        # Create two columns
    col1, col2 = st.columns(2)

    # Display pie chart in the first column
    with col1:
        st.subheader("WordCloud dari Judul Kekayaan Intelektual")
        generate_wordcloud(df_ki, "JUDUL")
    with col2:
        st.subheader("Daftar Kekayaan Intelektual Dengan Catatan")
        df_ki['CATATAN'] = df_ki['CATATAN'].replace(r'^\s*$', None, regex=True)
        df_filtered = df_ki[['JUDUL', 'CATATAN']]
        df_filtered = hapus_baris_kosong(df_filtered, 'CATATAN')
        st.dataframe(df_filtered, use_container_width=True)
        
    # Generate the network graph
    network_graph(df_ki_auth, df_ki_writer)

    # Menampilkan network graph dalam Streamlit
    st.subheader("Network Graph of Publications and Authors")
    # Membaca file HTML dengan encoding UTF-8
    with open("network_graph_w.html", "r", encoding="utf-8") as f:
        html_content = f.read()
        st.components.v1.html(html_content, height=800)
    # Menampilkan HTML di Streamlit

    # Menampilkan network graph dalam Streamlit
    st.subheader("Network Graph of Kelompok Riset")
    network(df_ki)
    # Membaca file HTML dengan encoding UTF-8
    with open("network.html", "r", encoding="utf-8") as f:
        html_graph = f.read()
        st.components.v1.html(html_graph, height=800)
        
if __name__ == "__main__":
    main()
