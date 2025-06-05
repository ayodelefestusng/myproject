import random
import datetime
from django.core.management.base import BaseCommand
from django.utils.timezone import make_aware
from faker import Faker
from myapp.models import (
    Customer, Supplier, Product, Sales, SalesItem,
    Employee, OnlineOrder, RetailOutlet, Inventory
)

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

    # --- MODEL GENERATION FUNCTIONS ---

    def create_outlets(self, n):
        """Generate Retail Outlets"""
        for _ in range(n):
            RetailOutlet.objects.create(
                name=fake.company(),
                state=fake.state_abbr(),
                city=fake.city(),
                address=fake.address(),
                online_flag=random.choice([True, False])
            )

    def create_customers(self, n):
        """Generate Customers"""
        for _ in range(n):
            Customer.objects.create(
                full_name=fake.name(),
                email=fake.email(),
                phone_number=fake.phone_number(),
                address=fake.address(),
                state=fake.state_abbr(),
                city=fake.city(),
                loyalty_points=random.randint(0, 1000)
            )

    def create_suppliers(self, n):
        """Generate Suppliers"""
        for _ in range(n):
            Supplier.objects.create(
                name=fake.company(),
                contact_info=fake.phone_number(),
                address=fake.address(),
                state=fake.state_abbr(),
                city=fake.city()
            )

    def create_products(self, n):
        """Generate Products"""
        categories = ["Electronics", "Groceries", "Fashion", "Automobile"]
        for _ in range(n):
            Product.objects.create(
                name=fake.word(),
                category=random.choice(categories),
                price=random.uniform(100, 50000),
                stock_quantity=random.randint(10, 500),
                description=fake.text()  # Added description field
            )

    # def create_inventory(self, n):
    #     """Generate Inventory Records"""
    #     products = list(Product.objects.all())
    #     for _ in range(n):
    #         if products:
    #             Inventory.objects.create(
    #                 product=random.choice(products),
    #                 quantity=random.randint(5, 200),
    #                 last_updated=make_aware(fake.date_time_this_year())
    #             )
        
    def create_inventory(self, n):
        """Generate Inventory Records"""
        products = list(Product.objects.all())
        outlets = list(RetailOutlet.objects.all())  # Ensure outlets exist

        if not outlets:
            self.stdout.write(self.style.ERROR("No RetailOutlet records found! Create some before running inventory generation."))
            return

        for _ in range(n):
            if products:
                Inventory.objects.create(
                    product=random.choice(products),
                    outlet=random.choice(outlets),
                    quantity=random.randint(5, 200)  # Ensure quantity is provided
                )         


    def create_employees(self, n):
        """Generate Employees"""
        outlets = list(RetailOutlet.objects.all())  # Ensure outlets exist

        if not outlets:
            self.stdout.write(self.style.ERROR("No RetailOutlet records found! Create some before running employee generation."))
            return

        roles = ["Cashier", "Sales Associate", "Manager", "Stock Clerk"]

        for _ in range(n):
            Employee.objects.create(
                full_name=fake.name(),
                role=random.choice(roles),
                outlet=random.choice(outlets),  # Ensures each employee belongs to an outlet
                salary=round(random.uniform(30000, 200000), 2),  # Generates a valid salary
                hire_date=fake.date_this_decade()  # Random hire date within the past 10 years
            )




    def create_online_orders(self, n):
        """Generate Online Orders"""
        customers = list(Customer.objects.all())

        if not customers:
            self.stdout.write(self.style.ERROR("No Customer records found! Create some before generating orders."))
            return

        payment_methods = ["Card", "Bank Transfer", "Digital Wallet"]
        order_statuses = ["Processing", "Shipped", "Delivered", "Returned"]

        for _ in range(n):
            OnlineOrder.objects.create(
                customer=random.choice(customers),
                payment_method=random.choice(payment_methods),
                delivery_address=fake.address(),
                order_status=random.choice(order_statuses),
                order_date=make_aware(fake.date_time_this_year())  # Ensures realistic timestamps
            )


    def create_sales(self, n):
        """Generate Sales Transactions"""
        outlets = list(RetailOutlet.objects.all())
        customers = list(Customer.objects.all())
        employees = list(Employee.objects.all())

        if not outlets:
            self.stdout.write(self.style.ERROR("No RetailOutlet records found! Create some before generating sales."))
            return

        payment_methods = ["Cash", "Card", "Online Transfer"]

        for _ in range(n):
            Sales.objects.create(
                outlet=random.choice(outlets),
                customer=random.choice(customers) if customers else None,
                employee=random.choice(employees) if employees else None,
                payment_method=random.choice(payment_methods),
                total_amount=round(random.uniform(500, 100000), 2),  # Ensures valid pricing
                sale_date=make_aware(fake.date_time_this_year())  # Creates realistic timestamps
            )


#  python manage.py generate_random_data --outlets 127 --customers 800 --suppliers 50 --products 50 --inventory 500 --employees 60 --sales 10000 --orders 2000


#  python manage.py generate_random_data  --orders 2000

