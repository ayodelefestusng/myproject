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

NIGERIAN_ADDRESS_FORMATS = [
    "{street}, {city}",
    "{building} {street}, {city}",
    "{number} {street}, {city}",
    "{street} Close, {city}",
    "{street} Avenue, {city}"
]

fake = Faker()

class Command(BaseCommand):
    help = "Generate sample Nigerian data for multiple models"

    def nigerian_name(self):
        """Generate Nigerian-style names"""
        first = random.choice(NIGERIAN_NAMES)
        last = random.choice(NIGERIAN_NAMES)
        return f"{first} {last}"

    def nigerian_state(self):
        """Return random Nigerian state code and name"""
        return random.choice(NIGERIAN_STATES)

    def nigerian_city(self, state_code):
        """Get random city for given state"""
        return random.choice(NIGERIAN_CITIES.get(state_code, ['Unknown']))

    def nigerian_address(self, city):
        """Generate Nigerian-style address"""
        street = fake.street_name()
        building = random.choice(['Block', 'House', 'Flat', 'Suite'])
        number = random.randint(1, 500)
        
        return random.choice(NIGERIAN_ADDRESS_FORMATS).format(
            street=street,
            city=city,
            building=building,
            number=number
        )

    def nigerian_phone(self):
        """Generate Nigerian phone number"""
        prefixes = ['080', '081', '070', '090', '091']
        return f"{random.choice(prefixes)}{random.randint(10000000, 99999999)}"

    def nigerian_company(self):
        """Generate Nigerian-style company names"""
        prefixes = ['Nigerian', 'Naija', 'Afri', 'West African', 'Lagos']
        suffixes = ['Ltd', 'PLC', 'Enterprises', 'Ventures', '& Sons']
        industry = random.choice(['Oil', 'Food', 'Tech', 'Fashion', 'Logistics'])
        return f"{random.choice(prefixes)} {industry} {random.choice(suffixes)}"

    def create_outlets(self, n):
        for _ in range(n):
            state_code, state_name = self.nigerian_state()
            city = self.nigerian_city(state_code)
            RetailOutlet.objects.create(
                name=self.nigerian_company(),
                state=state_code,
                city=city,
                address=self.nigerian_address(city),
                online_flag=random.choice([True, False])
            )

    def create_customers(self, n):
        for _ in range(n):
            state_code, state_name = self.nigerian_state()
            city = self.nigerian_city(state_code)
            Customer.objects.create(
                full_name=self.nigerian_name(),
                email=fake.email(),
                phone_number=self.nigerian_phone(),
                address=self.nigerian_address(city),
                state=state_code,
                city=city,
                loyalty_points=random.randint(0, 1000)
            )

    def create_suppliers(self, n):
        for _ in range(n):
            state_code, state_name = self.nigerian_state()
            city = self.nigerian_city(state_code)
            Supplier.objects.create(
                name=self.nigerian_company(),
                contact_info=self.nigerian_phone(),
                address=self.nigerian_address(city),
                state=state_code,
                city=city
            )

    # ... [rest of your methods remain the same, just update the calls to use the Nigerian-specific methods]

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting Nigerian data generation..."))

        self.create_outlets(options['outlets'])
        self.create_customers(options['customers'])
        self.create_suppliers(options['suppliers'])
        self.create_products(options['products'])
        self.create_inventory(options['inventory'])
        self.create_employees(options['employees'])
        self.create_sales(options['sales'])
        self.create_online_orders(options['orders'])

        self.stdout.write(self.style.SUCCESS(
            f"✅ Successfully generated Nigerian data:\n"
            f" - {options['outlets']} Retail Outlets\n"
            f" - {options['customers']} Customers\n"
            f" - {options['suppliers']} Suppliers\n"
            f" - {options['products']} Products\n"
            f" - {options['inventory']} Inventory records\n"
            f" - {options['employees']} Employees\n"
            f" - {options['sales']} Sales Transactions\n"
            f" - {options['orders']} Online Orders\n"
        ))