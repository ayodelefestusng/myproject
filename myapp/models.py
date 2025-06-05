from django.contrib.auth.models import AbstractUser, Group, Permission
from django.db import models
from django.contrib.auth.models import User
from django.db import models
from django.db import models
from django.contrib.auth.models import User

class Post(models.Model):
    title = models.CharField(max_length=150)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title


class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Conversation {self.id} with {self.user.username}"

# class Message(models.Model):
#     conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
#     text = models.TextField()
#     is_user = models.BooleanField()  # True for user, False for bot
#     timestamp = models.DateTimeField(auto_now_add=True)
    
#     class Meta:
#         ordering = ['timestamp']

#     def __str__(self):
#         sender = "User" if self.is_user else "Bot"
#         return f"{sender}: {self.text[:50]}..."
    
#     # No additional code needed here after removing duplicates.
    
    
    

class Message(models.Model):
    conversation = models.ForeignKey(Conversation, related_name='messages', on_delete=models.CASCADE)
    text = models.TextField()
    is_user = models.BooleanField()
    timestamp = models.DateTimeField(auto_now_add=True)
    attachment = models.FileField(upload_to='chat_attachments/', null=True, blank=True)

    class Meta:
        ordering = ['timestamp']

 
class LLM(models.Model):
	currecy_choices =[('NGN','Naira'),('GBP','Pound'),('EURO','Euro'),('USD','usd'),('DEDI','DEDI')]
	currency  =  models.CharField( max_length=50 ,choices=currecy_choices, default='NGN')
	naira = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
	pound =models.DecimalField( max_digits=50, decimal_places=2,default= 1)
	euro =models.DecimalField( max_digits=50, decimal_places=2,default= 1)
	usd = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
	cedi = models.DecimalField( max_digits=50, decimal_places=2,default= 1)
	
	
	def __str__(self):
		return self.currency
from django.db import models

class SessionData(models.Model):
    session_id = models.IntegerField(unique=True)
    sentiment = models.IntegerField()
    ticket = models.JSONField()  # Stores items as JSON

    def __str__(self):
        return f"Session {self.session_id}"

class Sentiment(models.Model):
    session_id = models.IntegerField(unique=True)
    sentimentperQuest = models.IntegerField()
    question = models.CharField(max_length=150)

    def __str__(self):
        return f"Session {self.session_id}"

class Insight(models.Model):
    # session_id = models.IntegerField(unique=False)
    session_id = models.CharField(max_length=150)
    sentimentAnswer = models.IntegerField(default=0, null=True, blank=True)
    answer = models.TextField()
    source = models.JSONField(blank=True, null=True)  # For storing list of strings
    ticket = models.JSONField(blank=True, null=True)  # For storing list of strings
    summary = models.TextField()
    sentiment = models.CharField(max_length=140)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True) 
    question = models.TextField(default="")
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, default=User.objects.get(username='aulagigo').id)
    # user = models.ForeignKey(User, on_delete=models.CASCADE)
    def __str__(self):
        return f"Insight {self.session_id}"

class Checkpoint(models.Model):
    thread_id = models.TextField()
    checkpoint_ns = models.TextField()
    data = models.BinaryField()
    

class Ticket(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Conversation {self.id} with {self.user.username}"
    
    
    
    
# Categories
CATEGORY_CHOICES = [
    ('Electronics', 'Electronics'), ('Clothing', 'Clothing'), ('Food', 'Food'),
    ('Health', 'Health'), ('Automobile', 'Automobile'), ('Books', 'Books'),
    ('Furniture', 'Furniture'), ('Beauty', 'Beauty'), ('Sports', 'Sports'), ('Toys', 'Toys')
]


# List of Nigerian States
NIGERIAN_STATES = [
    ('AB', 'Abia'), ('AD', 'Adamawa'), ('AK', 'Akwa Ibom'), ('AN', 'Anambra'), ('BA', 'Bauchi'),
    ('BE', 'Benue'), ('BO', 'Borno'), ('CR', 'Cross River'), ('DE', 'Delta'), ('EB', 'Ebonyi'),
    ('ED', 'Edo'), ('EK', 'Ekiti'), ('EN', 'Enugu'), ('GO', 'Gombe'), ('IM', 'Imo'), ('JI', 'Jigawa'),
    ('KD', 'Kaduna'), ('KN', 'Kano'), ('KT', 'Katsina'), ('KE', 'Kebbi'), ('KO', 'Kogi'), ('KW', 'Kwara'),
    ('LA', 'Lagos'), ('NA', 'Nasarawa'), ('NI', 'Niger'), ('OG', 'Ogun'), ('ON', 'Ondo'), ('OS', 'Osun'),
    ('OY', 'Oyo'), ('PL', 'Plateau'), ('RI', 'Rivers'), ('SO', 'Sokoto'), ('TA', 'Taraba'), ('YO', 'Yobe'), ('ZA', 'Zamfara')
]


CATEGORY_CHOICES = [
    ('Electronics', 'Electronics'), ('Clothing', 'Clothing'), ('Food', 'Food'),
    ('Health', 'Health'), ('Automobile', 'Automobile'), ('Books', 'Books'),
    ('Furniture', 'Furniture'), ('Beauty', 'Beauty'), ('Sports', 'Sports'), ('Toys', 'Toys')
]

class RetailOutlet(models.Model):
    name = models.CharField(max_length=150)
    state = models.CharField(max_length=150, choices=NIGERIAN_STATES)
    city = models.CharField(max_length=100)
    address = models.CharField(max_length=150)
    online_flag = models.BooleanField(default=False)

class Product(models.Model):
    name = models.CharField(max_length=150)
    category = models.CharField(max_length=100, choices=CATEGORY_CHOICES)
    price = models.DecimalField(max_digits=100, decimal_places=2)
    stock_quantity = models.IntegerField(default=0)  # Add this field with default value
    description = models.TextField()

class Inventory(models.Model):
    outlet = models.ForeignKey(RetailOutlet, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity  = models.IntegerField()
    last_updated = models.DateTimeField(auto_now=True)
    


# Define other models (Customer, Sales, Suppliers, Employee, OnlineOrders) similarly




class Customer(models.Model):
    full_name = models.CharField(max_length=150)
    email = models.EmailField(unique=False)
    phone_number = models.CharField(max_length=150)
    address = models.CharField(max_length=255)
    state = models.CharField(max_length=150, choices=NIGERIAN_STATES)
    city = models.CharField(max_length=100)
    loyalty_points = models.IntegerField(default=0)

    def __str__(self):
        return self.full_name

class Supplier(models.Model):
    name = models.CharField(max_length=255)
    contact_info = models.CharField(max_length=255)
    address = models.CharField(max_length=255)
    state = models.CharField(max_length=150, choices=NIGERIAN_STATES)
    city = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class Sales(models.Model):
    outlet = models.ForeignKey(RetailOutlet, on_delete=models.CASCADE)
    customer = models.ForeignKey(Customer, on_delete=models.SET_NULL, null=True, blank=True)
    employee = models.ForeignKey('Employee', on_delete=models.SET_NULL, null=True, blank=True)
    payment_method = models.CharField(max_length=150, choices=[
        ('Cash', 'Cash'), ('Card', 'Card'), ('Online Transfer', 'Online Transfer')
    ])
    total_amount = models.DecimalField(max_digits=10, decimal_places=2)
    sale_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Sale {self.id} at {self.outlet.name}"

class SalesItem(models.Model):
    sale = models.ForeignKey(Sales, on_delete=models.CASCADE)
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.PositiveIntegerField()
    unit_price = models.DecimalField(max_digits=10, decimal_places=2)

    def __str__(self):
        return f"{self.quantity} x {self.product.name}"




class Employee(models.Model):
    full_name = models.CharField(max_length=255)
    role = models.CharField(max_length=150, choices=[
        ('Cashier', 'Cashier'), ('Sales Associate', 'Sales Associate'),
        ('Manager', 'Manager'), ('Stock Clerk', 'Stock Clerk')
    ])
    outlet = models.ForeignKey(RetailOutlet, on_delete=models.CASCADE)
    salary = models.DecimalField(max_digits=10, decimal_places=2)
    hire_date = models.DateField()

    def __str__(self):
        return f"{self.full_name} - {self.role}"

class OnlineOrder(models.Model):
    customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    payment_method = models.CharField(max_length=150, choices=[
        ('Card', 'Card'), ('Bank Transfer', 'Bank Transfer'), ('Digital Wallet', 'Digital Wallet')
    ])
    delivery_address = models.CharField(max_length=255)
    order_status = models.CharField(max_length=150, choices=[
        ('Processing', 'Processing'), ('Shipped', 'Shipped'),
        ('Delivered', 'Delivered'), ('Returned', 'Returned')
    ])
    order_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Order {self.id} - {self.order_status}"
    
    


