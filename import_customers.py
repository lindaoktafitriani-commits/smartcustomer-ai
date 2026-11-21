
import os
import django
import csv

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

from customer_app.models import Customer

# Membaca file CSV
with open('Mall_Customers.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)

    jumlah_data = 0

    for row in reader:
        Customer.objects.update_or_create(
            customer_id=int(row['CustomerID']),
            defaults={
                'gender': row['Gender'],
                'age': int(row['Age']),
                'annual_income': int(row['Annual Income (k$)']),
                'spending_score': int(row['Spending Score (1-100)']),
            }
        )

        jumlah_data += 1

print(f"Berhasil mengimpor {jumlah_data} data pelanggan!")

# Menampilkan jumlah data dalam database
print(f"Total pelanggan dalam database: {Customer.objects.count()}")
