from django.shortcuts import render
from .models import Customer

import matplotlib
matplotlib.use('Agg')

from django.conf import settings
from groq import Groq

import matplotlib.pyplot as plt
import io
import base64
import markdown
import bleach

import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
from sklearn.preprocessing import StandardScaler

# 1. HALAMAN DASHBOARD DAN DATA PELANGGAN
def customer_list(request):
    customers = Customer.objects.all().order_by('customer_id')

    context = {
        'customers': customers,
        'total_customers': customers.count(),
    }

    return render(request, 'customer_app/customer_list.html', context)

# 2. HALAMAN ANALISIS DATA
def analysis(request):
    customers = list(Customer.objects.all())

    income = [c.annual_income for c in customers]
    spending = [c.spending_score for c in customers]

    grafik = None
    penjelasan_ai = None
    error_ai = None

    # Membuat grafik Matplotlib
    if customers:
        plt.figure(figsize=(9, 5))

        plt.scatter(
            income,
            spending,
            color='#2874cf',
            alpha=0.7,
            s=55
        )

        plt.title('Hubungan Annual Income dan Spending Score')
        plt.xlabel('Annual Income (k$)')
        plt.ylabel('Spending Score (1-100)')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        buffer.seek(0)

        grafik = base64.b64encode(
            buffer.getvalue()
        ).decode('utf-8')

        plt.close()

    # Meminta Groq menjelaskan grafik ketika tombol ditekan
    if request.method == 'POST':
        try:
            if not settings.GROQ_API_KEY:
                raise ValueError("API Key Groq belum tersedia.")

            if not customers:
                raise ValueError("Belum ada data pelanggan.")

            rata_income = np.mean(income)
            rata_spending = np.mean(spending)

            min_income = min(income)
            max_income = max(income)

            min_spending = min(spending)
            max_spending = max(spending)

            korelasi = np.corrcoef(income, spending)[0, 1]

            client = Groq(
                api_key=settings.GROQ_API_KEY
            )

            prompt = f"""
            Analisis grafik scatter plot pelanggan mall berikut.

            Jumlah pelanggan: {len(customers)}
            Rata-rata Annual Income: {rata_income:.2f} k$
            Rata-rata Spending Score: {rata_spending:.2f}
            Income terendah: {min_income} k$
            Income tertinggi: {max_income} k$
            Spending Score terendah: {min_spending}
            Spending Score tertinggi: {max_spending}
            Korelasi Pearson: {korelasi:.3f}

            Buat penjelasan dalam bahasa Indonesia yang mudah dipahami
            mahasiswa. Jelaskan:
            1. Gambaran umum grafik.
            2. Pola hubungan pendapatan dan pengeluaran.
            3. Temuan penting dari data.
            4. Kesimpulan dan manfaat bagi pengelola mall.

            Jangan mengarang pola yang tidak didukung angka.
            Jelaskan bahwa korelasi tidak selalu berarti sebab-akibat.
            Gunakan paragraf dan poin-poin yang rapi.
            """

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Kamu adalah asisten analis data "
                            "yang menjelaskan grafik secara akurat "
                            "dan mudah dipahami."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.4,
                max_completion_tokens=1000
            )

            penjelasan_ai = (
                response.choices[0].message.content
            )

            penjelasan_ai_html = markdown.markdown(
                penjelasan_ai,
                extensions=['tables', 'fenced_code']
            )

            penjelasan_ai_html = bleach.clean(
                penjelasan_ai_html,
                tags=[
                    'p', 'h1', 'h2', 'h3', 'h4',
                    'ul', 'ol', 'li',
                    'strong', 'em', 'blockquote',
                    'table', 'thead', 'tbody',
                    'tr', 'th', 'td', 'hr', 'br',
                    'pre', 'code'
                ],
                attributes={},
                strip=True
            )

        except Exception as e:
            error_ai = str(e)

    context = {
        'grafik': grafik,
        'total_customers': len(customers),
        'penjelasan_ai_html': penjelasan_ai_html if penjelasan_ai else None,
        'error_ai': error_ai,
    }

    return render(
        request,
        'customer_app/analysis.html',
        context
    )

# 3. SEGMENTASI FUZZY C-MEANS
def segmentasi(request):

    # Mengambil data pelanggan dari database
    customers = list(
        Customer.objects.all().order_by('customer_id')
    )

    if not customers:
        return render(request, 'customer_app/segmentasi.html', {
            'total_customers': 0,
            'grafik': None,
            'cluster_counts': {},
            'fpc': 0,
        })

    # Mengambil atribut pendapatan dan skor belanja
    data = np.array([
        [customer.annual_income, customer.spending_score]
        for customer in customers
    ], dtype=float)

    # Normalisasi data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Menentukan jumlah kelompok
    jumlah_cluster = 5

    # Menjalankan algoritma Fuzzy C-Means
    centers, membership, initial_membership, distance, objective, iterations, fpc = fuzz.cluster.cmeans(
        data=scaled_data.T,
        c=jumlah_cluster,
        m=2,
        error=0.005,
        maxiter=1000,
        seed=42
    )

    # Menentukan kelompok dominan setiap pelanggan
    labels = np.argmax(membership, axis=0) + 1

    # Menyimpan hasil segmentasi ke database
    for index, customer in enumerate(customers):
        customer.cluster_label = int(labels[index])

    Customer.objects.bulk_update(
        customers,
        ['cluster_label']
    )

    # Menghitung jumlah pelanggan setiap cluster
    cluster_counts = {}

    for cluster_number in range(1, jumlah_cluster + 1):
        cluster_counts[cluster_number] = int(
            np.sum(labels == cluster_number)
        )

    # Membuat grafik segmentasi
    plt.figure(figsize=(10, 6))

    for cluster_number in range(1, jumlah_cluster + 1):

        mask = labels == cluster_number

        plt.scatter(
            data[mask, 0],
            data[mask, 1],
            label=f'Cluster {cluster_number}',
            s=60,
            alpha=0.75
        )

    plt.title('Segmentasi Pelanggan Menggunakan Fuzzy C-Means')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Mengubah grafik menjadi gambar
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=120)
    buffer.seek(0)

    grafik = base64.b64encode(
        buffer.getvalue()
    ).decode('utf-8')

    plt.close()

    # Mengirim hasil ke halaman HTML
    context = {
        'grafik': grafik,
        'total_customers': len(customers),
        'cluster_counts': cluster_counts,
        'fpc': round(float(fpc), 4),
    }

    return render(
        request,
        'customer_app/segmentasi.html',
        context
    )

def fis_pelanggan(request):
    hasil = None
    error = None

    if request.method == 'POST':
        try:
            income = float(request.POST.get('annual_income'))
            spending = float(request.POST.get('spending_score'))

            if not 15 <= income <= 140:
                raise ValueError("Annual Income harus antara 15 sampai 140.")

            if not 1 <= spending <= 100:
                raise ValueError("Spending Score harus antara 1 sampai 100.")

            # 1. Membuat variabel input dan output
            pendapatan = ctrl.Antecedent(
                np.arange(15, 141, 1), 'pendapatan'
            )

            pengeluaran = ctrl.Antecedent(
                np.arange(1, 101, 1), 'pengeluaran'
            )

            potensi = ctrl.Consequent(
                np.arange(0, 101, 1), 'potensi'
            )

            # 2. Fungsi keanggotaan pendapatan
            pendapatan['rendah'] = fuzz.trimf(
                pendapatan.universe, [15, 15, 60]
            )
            pendapatan['sedang'] = fuzz.trimf(
                pendapatan.universe, [40, 80, 120]
            )
            pendapatan['tinggi'] = fuzz.trimf(
                pendapatan.universe, [100, 140, 140]
            )

            # 3. Fungsi keanggotaan pengeluaran
            pengeluaran['rendah'] = fuzz.trimf(
                pengeluaran.universe, [1, 1, 40]
            )
            pengeluaran['sedang'] = fuzz.trimf(
                pengeluaran.universe, [25, 50, 75]
            )
            pengeluaran['tinggi'] = fuzz.trimf(
                pengeluaran.universe, [60, 100, 100]
            )

            # 4. Fungsi keanggotaan hasil potensi
            potensi['rendah'] = fuzz.trimf(
                potensi.universe, [0, 0, 40]
            )
            potensi['sedang'] = fuzz.trimf(
                potensi.universe, [25, 50, 75]
            )
            potensi['tinggi'] = fuzz.trimf(
                potensi.universe, [60, 100, 100]
            )

            # 5. Membuat 9 aturan fuzzy
            aturan1 = ctrl.Rule(
                pendapatan['rendah'] & pengeluaran['rendah'],
                potensi['rendah']
            )
            aturan2 = ctrl.Rule(
                pendapatan['rendah'] & pengeluaran['sedang'],
                potensi['sedang']
            )
            aturan3 = ctrl.Rule(
                pendapatan['rendah'] & pengeluaran['tinggi'],
                potensi['sedang']
            )
            aturan4 = ctrl.Rule(
                pendapatan['sedang'] & pengeluaran['rendah'],
                potensi['rendah']
            )
            aturan5 = ctrl.Rule(
                pendapatan['sedang'] & pengeluaran['sedang'],
                potensi['sedang']
            )
            aturan6 = ctrl.Rule(
                pendapatan['sedang'] & pengeluaran['tinggi'],
                potensi['tinggi']
            )
            aturan7 = ctrl.Rule(
                pendapatan['tinggi'] & pengeluaran['rendah'],
                potensi['sedang']
            )
            aturan8 = ctrl.Rule(
                pendapatan['tinggi'] & pengeluaran['sedang'],
                potensi['tinggi']
            )
            aturan9 = ctrl.Rule(
                pendapatan['tinggi'] & pengeluaran['tinggi'],
                potensi['tinggi']
            )

            # 6. Menjalankan sistem fuzzy
            sistem = ctrl.ControlSystem([
                aturan1, aturan2, aturan3,
                aturan4, aturan5, aturan6,
                aturan7, aturan8, aturan9
            ])

            simulasi = ctrl.ControlSystemSimulation(sistem)

            simulasi.input['pendapatan'] = income
            simulasi.input['pengeluaran'] = spending
            simulasi.compute()

            nilai = simulasi.output['potensi']

            if nilai < 40:
                kategori = 'Rendah'
            elif nilai < 70:
                kategori = 'Sedang'
            else:
                kategori = 'Tinggi'

            hasil = {
                'income': income,
                'spending': spending,
                'nilai': round(float(nilai), 2),
                'kategori': kategori,
            }

        except (ValueError, TypeError) as e:
            error = str(e)
        except Exception:
            error = "Perhitungan FIS gagal. Periksa kembali input yang dimasukkan."

    return render(request, 'customer_app/fis.html', {
        'hasil': hasil,
        'error': error,
    })

