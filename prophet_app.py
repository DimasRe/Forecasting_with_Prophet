import streamlit as st
import pandas as pd
from prophet import Prophet

# Judul Aplikasi
st.title('Time Series Forecasting dengan Prophet')

st.write('Disini saya mau membuat aplikasi buat Forecasting.')

uploaded_file = st.file_uploader("Upload file CSV", type="CSV")

if uploaded_file:
    st.write('File Telah Ter-Upload')
    df = pd.read_csv(uploaded_file)
    st.write('Datafarme yang diupload:')
    st.write(df.shape)
    st.write(df.head())

    st.write("Pilih kolom untuk 'ds' (tanggal/waktu) dan 'y' (target/variable untuk forecasting)")
    col_date = st.selectbox("Pilih kolom untuk tanggal (ds)", options=df.columns)
    col_target = st.selectbox("Pilih kolom untuk target (y)", options=df.columns)

    if col_date == col_target:
        st.warning('Kolom tanggal dan target tidak boleh sama')
    else:
        df_prophet = df[[col_date, col_target]].rename(columns={col_date: 'ds', col_target: 'y'})
        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

        st.write("Data yang siap untuk Prophet:")
        st.write(df_prophet.head())

        period = st.slider("Pilih berapa hari untuk forecasting ke depan:", min_value=1, max_value=365)

        if st.button("Jalankan Forecasting"):
            model = Prophet()
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=period)
            forecast = model.predict(future)

            # Menampilkan Hasil Forecasting
            st.write("Forecasting Hasil:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            # Visualisasi Forecasting
            st.write("Grafik Forecasting:")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            st.write("Komponen Penting")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)
    
    