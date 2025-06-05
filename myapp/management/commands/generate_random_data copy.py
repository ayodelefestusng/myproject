import random
import datetime
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from faker import Faker
from myapp.models import (
    Customer, Supplier, Product, Sales, SalesItem,
    Employee, OnlineOrder, RetailOutlet, Inventory
)

# Nigerian-specific data
NIGERIAN_NAMES = [
    'Adebayo', 'Chinedu', 'Efe', 'Fatima', 'Gbenga', 'Halima', 'Ibrahim', 'Jumoke',
    'Kunle', 'Lola', 'Mohammed', 'Ngozi', 'Obinna', 'Patricia', 'Quadri', 'Rasheed',
    'Sade', 'Tunde', 'Uche', 'Yemi', 'Zainab'
]

NIGERIAN_STATES = [
    ('AB', 'Abia'), ('AD', 'Adamawa'), ('AK', 'Akwa Ibom'), ('AN', 'Anambra'),
    ('BA', 'Bauchi'), ('BE', 'Benue'), ('BO', 'Borno'), ('CR', 'Cross River'),
    ('DE', 'Delta'), ('EB', 'Ebonyi'), ('ED', 'Edo'), ('EK', 'Ekiti'),
    ('EN', 'Enugu'), ('GO', 'Gombe'), ('IM', 'Imo'), ('JI', 'Jigawa'),
    ('KD', 'Kaduna'), ('KN', 'Kano'), ('KT', 'Katsina'), ('KE', 'Kebbi'),
    ('KO', 'Kogi'), ('KW', 'Kwara'), ('LA', 'Lagos'), ('NA', 'Nasarawa'),
    ('NI', 'Niger'), ('OG', 'Ogun'), ('ON', 'Ondo'), ('OS', 'Osun'),
    ('OY', 'Oyo'), ('PL', 'Plateau'), ('RI', 'Rivers'), ('SO', 'Sokoto'),
    ('TA', 'Taraba'), ('YO', 'Yobe'), ('ZA', 'Zamfara')
]

NIGERIAN_CITIES = {
    'AB': ['Umuahia', 'Aba'], 'AD': ['Yola', 'Mubi'], 'AK': ['Uyo', 'Eket'],
    'AN': ['Awka', 'Onitsha'], 'BA': ['Bauchi', 'Azare'], 'BE': ['Makurdi', 'Gboko'],
    'BO': ['Maiduguri', 'Bama'], 'CR': ['Calabar', 'Ugep'], 'DE': ['Asaba', 'Warri'],
    'EB': ['Abakaliki', 'Afikpo'], 'ED': ['Benin City', 'Ekpoma'], 'EK': ['Ado Ekiti', 'Ikere'],
    'EN': ['Enugu', 'Nsukka'], 'GO': ['Gombe', 'Dukku'], 'IM': ['Owerri', 'Okigwe'],
    'JI': ['Dutse', 'Hadejia'], 'KD': ['Kaduna', 'Zaria'], 'KN': ['Kano', 'Dawakin Kudu'],
    'KT': ['Katsina', 'Daura'], 'KE': ['Birnin Kebbi', 'Argungu'], 'KO': ['Lokoja', 'Okene'],
    'KW': ['Ilorin', 'Offa'], 'LA': ['Lagos', 'Ikeja'], 'NA': ['Lafia', 'Keffi'],
    'NI': ['Minna', 'Bida'], 'OG': ['Abeokuta', 'Sagamu'], 'ON': ['Akure', 'Ondo'],
    'OS': ['Osogbo', 'Ilesa'], 'OY': ['Ibadan', 'Ogbomoso'], 'PL': ['Jos', 'Bukuru'],
    'RI': ['Port Harcourt', 'Bonny'], 'SO': ['Sokoto', 'Tambuwal'], 'TA': ['Jalingo', 'Wukari'],
    'YO': ['Damaturu', 'Potiskum'], 'ZA': ['Gusau', 'Kaura Namoda']
}

fake = Faker()

class Command(BaseCommand):
    help = "Generate sample Nigerian data for multiple models"

    def add_arguments(self, parser):
        parser.add_argument("--outlets", type=int, default=50, help="Number of retail outlets")
        parser.add_argument("--customers", type=int, default=200, help="Number of customers")
        parser.add_argument("--suppliers", type=int, default=20, help="Number of suppliers")
        parser.add_argument("--products", type=int, default=50, help="Number of products")
        parser.add_argument("--inventory", type=int, default=500, help="Number of inventory records")
        parser.add_argument("--employees", type=int, default=60, help="Number of employees")
        parser.add_argument("--sales", type=int, default=10000, help="Number of sales transactions")
        parser.add_argument("--orders", type=int, default=2000, help="Number of online orders")

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting Nigerian data generation..."))

        self.create_outlets(options["outlets"])
        self.create_customers(options["customers"])
        self.create_suppliers(options["suppliers"])
        self.create_products(options["products"])
        self.create_inventory(options["inventory"])
        self.create_employees(options["employees"])
        self.create_sales(options["sales"])
        self.create_online_orders(options["orders"])

        self.stdout.write(self.style.SUCCESS("✅ Successfully generated Nigerian data!"))

    def nigerian_name(self):
        """Generate Nigerian-style names"""
        first = random.choice(NIGERIAN_NAMES)
        last = random.choice(NIGERIAN_NAMES)
        return f"{first} {last}"

    def nigerian_state(self):
        """Return random Nigerian state"""
        return random.choice(NIGERIAN_STATES)

    def create_outlets(self, n):
        for _ in range(n):
            state_code, state_name = self.nigerian_state()
            city = random.choice(NIGERIAN_CITIES.get(state_code, ["Unknown"]))
            RetailOutlet.objects.create(
                name=fake.company(),
                state=state_code,
                city=city,
                address=fake.address(),
                online_flag=random.choice([True, False])
            )

    def create_customers(self, n):
        for _ in range(n):
            state_code, state_name = self.nigerian_state()
            city = random.choice(NIGERIAN_CITIES.get(state_code, ["Unknown"]))
            Customer.objects.create(
                full_name=self.nigerian_name(),
                email=fake.email(),
                phone_number=fake.phone_number(),
                address=fake.address(),
                state=state_code,
                city=city,
                loyalty_points=random.randint(0, 1000)
            )

    # Add similar methods for `create_suppliers`, `create_products`, etc.
# python manage.py generate_random_data --outlets 127 --customers 800 --suppliers 50 --products 50 --inventory 500 --employees 60 --sales 10000 --orders 2000