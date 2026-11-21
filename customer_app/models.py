from django.db import models


class Customer(models.Model):
    customer_id = models.PositiveIntegerField(primary_key=True)
    gender = models.CharField(max_length=10)
    age = models.PositiveIntegerField()
    annual_income = models.PositiveIntegerField()
    spending_score = models.PositiveIntegerField()

# Menyimpan hasil kelompok Fuzzy C-Means
    cluster_label = models.PositiveSmallIntegerField(
        null=True,
        blank=True
    )

    def __str__(self):
        return f"Customer {self.customer_id}"
