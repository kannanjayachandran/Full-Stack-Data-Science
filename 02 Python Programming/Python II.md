<h1 align="center" > Python Programming - II </h1>

---

## Table of Contents

1. **Object-Oriented Programming (OOP)**
    - [Core OOP Principles](#core-oop-principles)
    - [Class and Object](#class-and-object)
    - [Methods](#methods)
    - [Magic/Dunder methods](#magic-methods-or-special-methods)
    - [Inheritance and Method Overriding](#inheritance-and-method-overriding)
    - [Multiple Inheritance](#multiple-inheritance)
    - [Method Resolution Order (MRO)](#method-resolution-order-mro)
    - [Encapsulation](#encapsulation)
    - [`@property` decorator](#properties-and-descriptors)
    - [Abstraction and `abc` module](#abstract-base-classes)
    - [Virtual Subclasses](#virtual-subclasses)
    - [Metaclasses and Dynamic Class Creation](#metaclasses-and-dynamic-class-creation)
    - [Metaclasses vs Class Decorators](#metaclass-vs-class-decorators)

2. **Functional Programming Concepts**
    - [FP vs OOP](#fp-vs-oop)
    - [Core Principles of FP](#core-principles-of-functional-programming)
    - [Higher-order Functions](#higher-order-functions)
    - [`map()`, `filter()`, `reduce()`, `zip()`, `enumerate()`](#built-in-functional-tools)
    - [Lambda Functions](#lambda-functions)
    - [Closures](#closures)
    - [Decorators](#decorators)
    - [Generators](#generators)
    - [Iterators and the Iterator Protocol](#iterators-and-the-iterator-protocol)
    - [Advanced Functional Patterns (Partial application, Currying, Lazy evaluation patterns, Monads, Functional Data Structures)](#advanced-functional-patterns)
    - [Example FP codes](#example-codes)
    - [FP Patterns](#common-patterns)
    - [Testing FP code](#testing-functional-code)

---

## Object-Oriented Programming (OOP) in Python

Object-Oriented Programming (OOP) is a programming paradigm based on the concept of *objects*, which encapsulate data (attributes) and behavior (methods).

In Python, **everything is an object** — this includes integers, strings, functions, classes, and even modules. Each object is an instance of a class and has:

- **Identity**: Unique ID (obtained via `id()`)
- Type: The class it belongs to (obtained via `type()`)
- Value: The data it holds

```py
x = 42
print(type(x))      # <class 'int'>
print(isinstance(x, object))    # True

def my_func():
    pass

print(type(my_func))     # <class 'function'>
print(isinstance(my_func, object))  # True

# Even classes are objects (instances of 'type')
class MyClass:
    pass

print(type(MyClass))     # <class 'type'>
print(isinstance(MyClass, object))  # True
```

### Core OOP Principles

- **Encapsulation**: Bundling data and methods that operate on that data within a single unit (class)
- **Abstraction**: Hiding complex implementation details and exposing only essential features
- **Inheritance**: Creating new classes from existing ones to promote code reuse
- **Polymorphism**: Ability to use a common interface for different underlying data types

### Class and Object

- **Class:** A blueprint or template that defines the structure and behavior of objects.
- **Object:** An instance of a class containing real data.

```python
class Dog:
    """A simple Dog class demonstrating basic OOP concepts"""
    
    # Class attribute (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        """Instance initializer (constructor)"""
        # Instance attributes (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        """Instance method"""
        return f"{self.name} says Woof!"

# Creating objects (instances)
dog1 = Dog("Buddy", 3)
dog2 = Dog("Lucy", 5)

print(dog1.name)          # Buddy
print(dog1.species)       # Canis familiaris
print(dog1.bark())        # Buddy says Woof!

# Class attributes are shared
print(dog1.species is dog2.species)  # True (same object in memory)
```

#### Instance vs Class Attributes

- Class attributes are shared across all instances
- Instance attributes are unique to each object
- Accessing a class attribute through an instance creates a lookup chain: `instance → class`
- Assigning to an attribute through an instance creates an instance attribute (doesn't modify class attribute)

Functions defined inside a class are known as **methods**. 

- *classes* in Python are **callable objects** that return instances.

### Methods

Methods are Functions defined inside a class. They can access and modify the data associated with the object.

#### Types of Methods in Python OOP

- **Instance Methods**

Operate on instance data and have access to instance data (`self`).

```py
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        """Instance method - modifies instance state"""
        if amount > 0:
            self.balance += amount
            return f"Deposited ${amount}. New balance: ${self.balance}"
        return "Invalid amount"
    
    def withdraw(self, amount):
        """Instance method with validation"""
        if 0 < amount <= self.balance:
            self.balance -= amount
            return f"Withdrew ${amount}. New balance: ${self.balance}"
        return "Insufficient funds or invalid amount"

account = BankAccount("Alice", 1000)
print(account.deposit(500))   # Deposited $500. New balance: $1500
print(account.withdraw(200))  # Withdrew $200. New balance: $1300
```

- Instance method can acutally access class data via `self.__class__`

- **Class Methods**

Operate on the class data, not instances. Use the `@classmethod` decorator and `cls` as the first parameter. Often used as **factory methods** to create objects in alternative ways, or to access and modify class variables.

```py
class Employee:
    company = "TechCorp"
    employee_count = 0
    
    def __init__(self, name, salary):
        self.name = name
        self.salary = salary
        Employee.employee_count += 1
    
    @classmethod
    def get_employee_count(cls):
        """Class method - accesses class attributes"""
        return f"{cls.company} has {cls.employee_count} employees"
    
    @classmethod
    def from_string(cls, emp_string):
        """Alternative constructor pattern"""
        name, salary = emp_string.split('-')
        return cls(name, int(salary))
    
    @classmethod
    def set_company_name(cls, name):
        """Modify class attribute"""
        cls.company = name

# Using class methods
emp1 = Employee("Alice", 50000)
emp2 = Employee("Bob", 60000)

print(Employee.get_employee_count())  # TechCorp has 2 employees

# Alternative constructor
emp3 = Employee.from_string("Charlie-55000")
print(emp3.name)  # Charlie

# Modifying class attribute
Employee.set_company_name("NewTech")
print(Employee.company)  # NewTech
```

- **Static Methods**

Behave like normal functions but belong to the class’s namespace. Defined with `@staticmethod`.

```py
class MathOperations:
    """Collection of mathematical utilities"""
    
    @staticmethod
    def add(x, y):
        """Static method - no access to instance or class"""
        return x + y
    
    @staticmethod
    def is_even(num):
        """Pure utility function"""
        return num % 2 == 0
    
    @staticmethod
    def validate_positive(value):
        """Validation helper"""
        if value <= 0:
            raise ValueError("Value must be positive")
        return True

# Using static methods (no instance needed)
print(MathOperations.add(5, 3))        # 8
print(MathOperations.is_even(10))      # True

# Can also call through instance (but not recommended)
math = MathOperations()
print(math.add(2, 3))                  # 5
```

- Use **class methods for factory patterns** and **static methods for utility/helper functions**.

### Magic Methods or Special Methods

Magic Methods or Special Methods or Dunder Methods have predefined meanings and are invoked implicitly by Python in specific situations.

- **`__init__`** :  It is called immediately after object creation. It's used to initialize instance attributes
    - `__init__` is not a constructor, it's an initializer
    - `self` refers to the instance being initialized
    - `__new__` is the constructor and its rarely overridden

```py
class Person:
    def __init__(self, name, age=0):
        """Initialize person with validation"""
        if not isinstance(name, str):
            raise TypeError("Name must be a string")
        if age < 0:
            raise ValueError("Age cannot be negative")
        
        self.name = name
        self.age = age

# Valid initialization
person1 = Person("Alice", 30)

# Using default argument
person2 = Person("Bob")

# Invalid initialization
try:
    person3 = Person(123)  # TypeError
except TypeError as e:
    print(f"Error: {e}")
```


- **`__str__`** and **`__repr__`** (String representations) : `__str__` is the human-readable representation for end users, while, `__repr__` is the unambiguous representation for developers (ideally, evaluating it should recreate the object)
    - Always implement `__repr__`; implement `__str__` only if a different user-facing representation is needed
    - Use `!r` in f-strings within `__repr__` to get `repr()` of attributes

```py
class Book:
    def __init__(self, title, author, year):
        self.title = title
        self.author = author
        self.year = year
    
    def __str__(self):
        """User-friendly string representation"""
        return f'"{self.title}" by {self.author}'
    
    def __repr__(self):
        """Developer-friendly representation (should be unambiguous)"""
        return f'Book(title={self.title!r}, author={self.author!r}, year={self.year})'

book = Book("1984", "George Orwell", 1949)

# __str__ is called by print() and str()
print(str(book))          # "1984" by George Orwell
print(book)               # "1984" by George Orwell

# __repr__ is called by repr() and in interactive shell
print(repr(book))         # Book(title='1984', author='George Orwell', year=1949)

# If __str__ is not defined, __repr__ is used as fallback
```

| Category       | Method                                  | Description                            |
| -------------- | --------------------------------------- | -------------------------------------- |
| Initialization | `__init__`, `__new__`                   | Object creation and initialization     |
| Representation | `__str__`, `__repr__`                   | String representations                 |
| Arithmetic     | `__add__`, `__sub__`, `__mul__`         | Operator overloading                   |
| Comparison     | `__eq__`, `__lt__`, `__gt__`            | Custom comparisons                     |
| Container      | `__getitem__`, `__setitem__`, `__len__` | Collection-like behavior               |
| Callable       | `__call__`                              | Make instances callable like functions |

- Python doesn't support traditional **constructor overloading**, but provides alternatives:

```py
class Rectangle:
    def __init__(self, width=None, height=None, square_side=None):
        """Flexible initialization"""
        if square_side is not None:
            self.width = self.height = square_side
        elif width is not None and height is not None:
            self.width = width
            self.height = height
        else:
            raise ValueError("Provide either width and height, or square_side")
    
    @classmethod
    def from_square(cls, side):
        """Alternative constructor for squares"""
        return cls(width=side, height=side)
    
    @classmethod
    def from_dimensions(cls, dimensions):
        """Create from tuple or list"""
        width, height = dimensions
        return cls(width=width, height=height)
    
    def area(self):
        return self.width * self.height

# Different ways to create rectangles
rect1 = Rectangle(10, 20)
rect2 = Rectangle(square_side=15)
rect3 = Rectangle.from_square(15)
rect4 = Rectangle.from_dimensions((10, 20))

print(rect1.area())  # 200
print(rect2.area())  # 225
```

- Use `@classmethod` for alternative constructors
- Provides clear, named entry points for different initialization scenarios

### Inheritance and Method Overriding

Inheritance allows a class (child) to acquire properties and methods of another class (parent).

```py
class Animal:
    """Base class"""
    
    def __init__(self, name, species):
        self.name = name
        self.species = species
    
    def make_sound(self):
        """Method to be overridden"""
        return "Some generic sound"
    
    def info(self):
        """Inherited method"""
        return f"{self.name} is a {self.species}"

class Dog(Animal):
    """Derived class"""
    
    def __init__(self, name, breed):
        # Call parent constructor
        super().__init__(name, species="Dog")
        self.breed = breed
    
    def make_sound(self):
        """Method overriding"""
        return "Woof!"
    
    def fetch(self):
        """New method specific to Dog"""
        return f"{self.name} is fetching the ball"

class Cat(Animal):
    """Another derived class"""
    
    def __init__(self, name, indoor=True):
        super().__init__(name, species="Cat")
        self.indoor = indoor
    
    def make_sound(self):
        """Method overriding"""
        return "Meow!"

# Using inheritance
dog = Dog("Buddy", "Golden Retriever")
cat = Cat("Whiskers", indoor=True)

print(dog.info())         # Buddy is a Dog (inherited method)
print(dog.make_sound())   # Woof! (overridden method)
print(dog.fetch())        # Buddy is fetching the ball (new method)

print(cat.info())         # Whiskers is a Cat
print(cat.make_sound())   # Meow!

# Type checking
print(isinstance(dog, Dog))     # True
print(isinstance(dog, Animal))  # True (Dog is subclass of Animal)
print(isinstance(dog, Cat))     # False
print(issubclass(Dog, Animal))  # True
```

#### Method overriding rules

```py
class Parent:
    def method(self):
        return "Parent method"
    
    def another_method(self):
        return "Parent another_method"

class Child(Parent):
    def method(self):
        """Complete override"""
        return "Child method"
    
    def another_method(self):
        """Extending parent behavior"""
        parent_result = super().another_method()
        return f"{parent_result} + Child extension"

child = Child()
print(child.method())          # Child method
print(child.another_method())  # Parent another_method + Child extension
```

#### `super()` function

`super()` provides access to methods in parent classes, respecting the Method Resolution Order (MRO).

```py
class Shape:
    def __init__(self, color):
        self.color = color
        print(f"Shape.__init__ called with color={color}")
    
    def area(self):
        raise NotImplementedError("Subclass must implement area()")

class Rectangle(Shape):
    def __init__(self, color, width, height):
        print(f"Rectangle.__init__ called")
        super().__init__(color)  # Call parent initializer
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Square(Rectangle):
    def __init__(self, color, side):
        print(f"Square.__init__ called")
        super().__init__(color, side, side)  # Call Rectangle's initializer

# Tracing initialization chain
square = Square("red", 5)
# Output:
# Square.__init__ called
# Rectangle.__init__ called
# Shape.__init__ called with color=red

print(square.area())  # 25
```

- In **Python 3**, `super()` without arguments is equivalent to `super(CurrentClass, self)`
- Follows **MRO**, not just immediate parent
- Essential for **cooperative multiple inheritance**
- Can be called anywhere in a method, not just at the beginning
- Python does not support traditional *overloading*, but provides alternatives:

```py
class Calculator:
    def add(self, a, b=None, c=None):
        """Using default arguments"""
        if b is None:
            return a
        if c is None:
            return a + b
        return a + b + c
    
    def multiply(self, *args):
        """Using variable arguments"""
        result = 1
        for num in args:
            result *= num
        return result
    
    def process(self, data):
        """Using type checking (discouraged in Python)"""
        if isinstance(data, int):
            return data * 2
        elif isinstance(data, str):
            return data.upper()
        elif isinstance(data, list):
            return sum(data)
        else:
            raise TypeError(f"Unsupported type: {type(data)}")

calc = Calculator()
print(calc.add(5))           # 5
print(calc.add(5, 3))        # 8
print(calc.add(5, 3, 2))     # 10

print(calc.multiply(2, 3, 4))     # 24
print(calc.process(5))            # 10
print(calc.process("hello"))      # HELLO
print(calc.process([1, 2, 3]))    # 6
```

- **Pythonic Alternatives**:
    - Default arguments
    - Variable-length arguments (`*args`, `**kwargs`)
    - Single dispatch (from `functools.singledispatch` for function-based approach)
    - Duck typing (accept any object with required methods)

#### Multiple Inheritance

Multiple inheritance allows a class to inherit from more than one parent class.

```py
class Flyable:
    """Mixin for flying capability"""
    
    def fly(self):
        return f"{self.name} is flying"

class Swimmable:
    """Mixin for swimming capability"""
    
    def swim(self):
        return f"{self.name} is swimming"

class Animal:
    """Base animal class"""
    
    def __init__(self, name):
        self.name = name
    
    def eat(self):
        return f"{self.name} is eating"

class Duck(Animal, Flyable, Swimmable):
    """Duck inherits from three classes"""
    
    def __init__(self, name):
        super().__init__(name)
    
    def quack(self):
        return f"{self.name} says Quack!"

# Using multiple inheritance
duck = Duck("Donald")
print(duck.eat())     # Donald is eating (from Animal)
print(duck.fly())     # Donald is flying (from Flyable)
print(duck.swim())    # Donald is swimming (from Swimmable)
print(duck.quack())   # Donald says Quack! (from Duck)
```

### Method Resolution Order (MRO)

MRO defines the order in which Python searches for methods in the inheritance hierarchy. Python uses **C3 Linearization** algorithm to compute MRO.

```py
class A:
    def process(self):
        return "A"

class B(A):
    def process(self):
        return "B"

class C(A):
    def process(self):
        return "C"

class D(B, C):
    """D inherits from both B and C"""
    pass

# Checking MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

print(D.mro())  # Same as above, but as a list
# [<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>]

# Method resolution follows MRO
d = D()
print(d.process())  # "B" (found in B first, not C)
```

- **MRO Visualization**:

![Standard inheritance hierarchy](./img/Inheritance_hierarchy.png)

![MRO Order](./img/MRO.png)

> Python resolves methods **left-to-right**, **depth-first**, **skipping duplicates**.

#### Diamond Problem

The diamond problem occurs when a class inherits from two classes that share a common ancestor.

```py
class Animal:
    def __init__(self):
        print("Animal.__init__")
        self.animal_initialized = True

class Mammal(Animal):
    def __init__(self):
        print("Mammal.__init__")
        super().__init__()  # Calls next in MRO
        self.mammal_initialized = True

class Bird(Animal):
    def __init__(self):
        print("Bird.__init__")
        super().__init__()  # Calls next in MRO
        self.bird_initialized = True

class Bat(Mammal, Bird):
    """Diamond inheritance: Bat -> Mammal -> Animal
                             Bat -> Bird -> Animal"""
    def __init__(self):
        print("Bat.__init__")
        super().__init__()  # Follows MRO
        self.bat_initialized = True

# Creating a Bat
bat = Bat()
# Output:
# Bat.__init__
# Mammal.__init__
# Bird.__init__
# Animal.__init__

# Check MRO
print(Bat.__mro__)
# (<class 'Bat'>, <class 'Mammal'>, <class 'Bird'>, <class 'Animal'>, <class 'object'>)

# Animal.__init__ is called only once (no duplicate initialization)
print(hasattr(bat, 'animal_initialized'))  # True
print(hasattr(bat, 'mammal_initialized'))  # True
print(hasattr(bat, 'bird_initialized'))    # True
print(hasattr(bat, 'bat_initialized'))     # True
```

- **C3 Linearization** ensures each class appears only once in MRO
- `super()` calls the next class in MRO, not necessarily the parent
- Common ancestor (Animal) is initialized only once
- All `__init__` methods should call `super().__init__()` for cooperative multiple inheritance

#### Cooperative Multiple Inheritance

```py
class LoggingMixin:
    """Mixin that adds logging capability"""
    
    def __init__(self, *args, **kwargs):
        print(f"LoggingMixin.__init__ called")
        super().__init__(*args, **kwargs)  # Pass along to next in MRO
        self.logs = []
    
    def log(self, message):
        self.logs.append(message)

class ValidationMixin:
    """Mixin that adds validation"""
    
    def __init__(self, *args, **kwargs):
        print(f"ValidationMixin.__init__ called")
        super().__init__(*args, **kwargs)
        self.validated = True
    
    def validate(self):
        return self.validated

class DataModel:
    """Base data model"""
    
    def __init__(self, data):
        print(f"DataModel.__init__ called with data={data}")
        self.data = data

class EnhancedModel(LoggingMixin, ValidationMixin, DataModel):
    """Model with logging and validation"""
    
    def __init__(self, data):
        print(f"EnhancedModel.__init__ called")
        super().__init__(data)
        self.log("Model created")

# Creating enhanced model
model = EnhancedModel({"key": "value"})
# Output:
# EnhancedModel.__init__ called
# LoggingMixin.__init__ called
# ValidationMixin.__init__ called
# DataModel.__init__ called with data={'key': 'value'}

print(EnhancedModel.__mro__)
# (<class 'EnhancedModel'>, <class 'LoggingMixin'>, 
#  <class 'ValidationMixin'>, <class 'DataModel'>, <class 'object'>)

# All capabilities available
print(model.data)           # {'key': 'value'}
print(model.validate())     # True
print(model.logs)           # ['Model created']
```

- Always use `super()` for cooperative inheritance
- Pass `*args`, `**kwargs` through `super().__init__()` calls
- Use **mixins** for composable functionality (single-purpose classes)
- Avoid **state** in mixins when possible
- Document MRO for complex hierarchies
- Check MRO with `ClassName.__mro__` or `ClassName.mro()`

#### Mixins

Mixins are small, reusable classes designed to add specific functionality to other classes.

```py
class JSONMixin:
    """Mixin to add JSON serialization"""
    
    def to_json(self):
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_string):
        import json
        data = json.loads(json_string)
        return cls(**data)

class ComparableMixin:
    """Mixin for comparison based on 'value' attribute"""
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value < other.value
    
    def __le__(self, other):
        return self.value <= other.value

class Product(JSONMixin, ComparableMixin):
    """Product with JSON and comparison capabilities"""
    
    def __init__(self, name, value):
        self.name = name
        self.value = value
    
    def __repr__(self):
        return f"Product({self.name}, ${self.value})"

# Using mixins
p1 = Product("Laptop", 1000)
p2 = Product("Mouse", 50)

# JSON serialization (from JSONMixin)
json_str = p1.to_json()
print(json_str)  # {"name": "Laptop", "value": 1000}

p3 = Product.from_json(json_str)
print(p3)  # Product(Laptop, $1000)

# Comparison (from ComparableMixin)
print(p1 > p2)  # True
print(p1 == p3)  # True
print(sorted([p1, p2]))  # [Product(Mouse, $50), Product(Laptop, $1000)]
```

> Use mixins to add small, reusable behaviors to classes — avoid deep inheritance chains.

### Encapsulation

Encapsulation is the process of bundling data and methods together. Python uses naming conventions rather than true access modifiers (like `private` in Java/C++).

- `_single_underscore` → **protected** (convention only)

- `__double_underscore` → name mangling for **private** attributes

```py
class Account:
    def __init__(self, owner, balance):
        self.owner = owner           # Public attribute
        self._balance = balance      # Protected (convention: internal use)
        self.__pin = "1234"          # Private (name mangling applied)
    
    def get_balance(self):
        """Public method to access protected attribute"""
        return self._balance
    
    def __validate_pin(self, pin):
        """Private method (name mangling)"""
        return pin == self.__pin
    
    def withdraw(self, amount, pin):
        """Public method using private validation"""
        if self.__validate_pin(pin):
            if amount <= self._balance:
                self._balance -= amount
                return f"Withdrew ${amount}"
            return "Insufficient funds"
        return "Invalid PIN"

account = Account("Alice", 1000)

# Public access
print(account.owner)  # Alice

# Protected access (nothing prevents it, just convention)
print(account._balance)  # 1000 (not recommended, but possible)

# Private access (name mangling)
try:
    print(account.__pin)  # AttributeError
except AttributeError as e:
    print(f"Error: {e}")

# Private attributes are mangled to _ClassName__attribute
print(account._Account__pin)  # 1234 (not truly private, but obfuscated)

# Using public interface
print(account.withdraw(200, "1234"))  # Withdrew $200
print(account.get_balance())           # 800
```

- **Naming Conventions Summary**

```py
class Example:
    def __init__(self):
        self.public = "I'm public"
        self._protected = "I'm protected by convention"
        self.__private = "I'm name-mangled"
    
    def public_method(self):
        return "Anyone can call me"
    
    def _protected_method(self):
        return "Internal use suggested"
    
    def __private_method(self):
        return "Name mangled method"
    
    def access_private(self):
        """Public interface to private members"""
        return self.__private_method()

obj = Example()

# All are technically accessible
print(obj.public)                    # Works
print(obj._protected)                # Works (not recommended)
print(obj._Example__private)         # Works (name mangling exposed)

# Methods
print(obj.public_method())           # Works
print(obj._protected_method())       # Works (not recommended)
print(obj.access_private())          # Works (proper way)
```

- **Best Practices for Encapsulation**

```py
class BankAccount:
    """Demonstrating proper encapsulation"""
    
    def __init__(self, account_number, initial_balance):
        self.__account_number = account_number  # Private
        self.__balance = initial_balance        # Private
        self._transaction_history = []          # Protected
    
    @property
    def account_number(self):
        """Read-only access to account number"""
        return f"****{self.__account_number[-4:]}"  # Masked
    
    @property
    def balance(self):
        """Read-only access to balance"""
        return self.__balance
    
    def deposit(self, amount):
        """Public method with validation"""
        if amount > 0:
            self.__balance += amount
            self._transaction_history.append(f"Deposit: +${amount}")
            return True
        raise ValueError("Deposit amount must be positive")
    
    def withdraw(self, amount):
        """Public method with business logic"""
        if amount > 0 and amount <= self.__balance:
            self.__balance -= amount
            self._transaction_history.append(f"Withdrawal: -${amount}")
            return True
        raise ValueError("Invalid withdrawal amount")
    
    def _calculate_interest(self, rate):
        """Protected helper method"""
        return self.__balance * rate
    
    def __str__(self):
        return f"Account {self.account_number}: ${self.balance}"

# Usage
account = BankAccount("123456789", 1000)
print(account)                    # Account ****6789: $1000
account.deposit(500)
print(account.balance)            # 1500

# Cannot directly modify balance
# account.balance = 5000  # AttributeError (if using @property without setter)
```

#### Properties and Descriptors

- **`@property` Decorator**

Properties provide a Pythonic way to implement **getters**, **setters**, and **deleters** while maintaining attribute-like syntax.

```py
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius  # Private attribute
    
    @property
    def celsius(self):
        """Getter for celsius"""
        print("Getting celsius")
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Setter with validation"""
        print(f"Setting celsius to {value}")
        if value < -273.15:
            raise ValueError("Temperature below absolute zero!")
        self._celsius = value
    
    @celsius.deleter
    def celsius(self):
        """Deleter"""
        print("Deleting celsius")
        del self._celsius
    
    @property
    def fahrenheit(self):
        """Computed property (read-only)"""
        return (self.celsius * 9/5) + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Setting fahrenheit updates celsius"""
        self.celsius = (value - 32) * 5/9

# Using properties
temp = Temperature(25)
print(temp.celsius)        # Getting celsius → 25
print(temp.fahrenheit)     # Getting celsius → 77.0

temp.celsius = 30          # Setting celsius to 30
print(temp.celsius)        # Getting celsius → 30

temp.fahrenheit = 86       # Setting celsius to 30.0 (via fahrenheit setter)
print(temp.celsius)        # Getting celsius → 30.0

# Validation works
try:
    temp.celsius = -300    # ValueError
except ValueError as e:
    print(e)

# Deletion
del temp.celsius           # Deleting celsius
```

- **Property vs Direct Attribute Access**

```py
# Without property (not recommended for complex logic)
class CircleBasic:
    def __init__(self, radius):
        self.radius = radius
        self.diameter = radius * 2  # Problem: can become inconsistent
    
    def area(self):
        return 3.14159 * self.radius ** 2

circle1 = CircleBasic(5)
print(circle1.diameter)  # 10
circle1.radius = 10      # Changing radius
print(circle1.diameter)  # Still 10 (inconsistent!)

# With property (recommended)
class CircleProperty:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value <= 0:
            raise ValueError("Radius must be positive")
        self._radius = value
    
    @property
    def diameter(self):
        """Computed property - always consistent"""
        return self._radius * 2
    
    @diameter.setter
    def diameter(self, value):
        self._radius = value / 2
    
    @property
    def area(self):
        """Read-only computed property"""
        return 3.14159 * self._radius ** 2

circle2 = CircleProperty(5)
print(circle2.diameter)  # 10
circle2.radius = 10      # Changing radius
print(circle2.diameter)  # 20 (consistent!)
print(circle2.area)      # 314.159

circle2.diameter = 30    # Setting via diameter
print(circle2.radius)    # 15.0
```

- **Custom Descriptors**

Descriptors are the mechanism behind properties. They allow you to customize attribute access across multiple classes.

```py
class ValidatedString:
    """Descriptor for validated string attributes"""
    
    def __init__(self, minlen=0, maxlen=100):
        self.minlen = minlen
        self.maxlen = maxlen
    
    def __set_name__(self, owner, name):
        """Called when descriptor is assigned to a class attribute"""
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        """Called when attribute is accessed"""
        if instance is None:
            return self  # Accessing from class, not instance
        return getattr(instance, self.name, None)
    
    def __set__(self, instance, value):
        """Called when attribute is set"""
        if not isinstance(value, str):
            raise TypeError(f"{self.name} must be a string")
        if len(value) < self.minlen:
            raise ValueError(f"{self.name} must be at least {self.minlen} characters")
        if len(value) > self.maxlen:
            raise ValueError(f"{self.name} must be at most {self.maxlen} characters")
        setattr(instance, self.name, value)

class PositiveNumber:
    """Descriptor for positive numbers"""
    
    def __set_name__(self, owner, name):
        self.name = f"_{name}"
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.name, 0)
    
    def __set__(self, instance, value):
        if not isinstance(value, (int, float)):
            raise TypeError(f"{self.name} must be a number")
        if value <= 0:
            raise ValueError(f"{self.name} must be positive")
        setattr(instance, self.name, value)

class Product:
    """Using descriptors for validation"""
    name = ValidatedString(minlen=3, maxlen=50)
    price = PositiveNumber()
    
    def __init__(self, name, price):
        self.name = name    # Calls ValidatedString.__set__
        self.price = price  # Calls PositiveNumber.__set__

# Using the descriptor-based class
product = Product("Laptop", 999.99)
print(product.name)   # Laptop
print(product.price)  # 999.99

# Validation works automatically
try:
    product.name = "AB"  # Too short
except ValueError as e:
    print(e)

try:
    product.price = -10  # Negative
except ValueError as e:
    print(e)

try:
    product.price = "expensive"  # Wrong type
except TypeError as e:
    print(e)
```

- **When to use descriptors**:
    - Reusable validation logic across multiple classes
    - Complex attribute management (caching, type checking, etc.)
    - Framework/library development
    - Note: For simple cases, @property is usually sufficient

#### Benefits of Encapsulation

- **Data validation**: Control how attributes are modified
- **Internal representation freedom**: Change implementation without affecting external code
- **Prevents accidental modification**: Reduces bugs from unintended state changes
- **Clear interface**: Public methods define how objects should be used

### Abstract Base Classes

Abstract Base Classes (ABCs) define a contract that subclasses must implement. They cannot be instantiated directly.

```py
from abc import ABC, abstractmethod

class Shape(ABC):
    """Abstract base class for shapes"""
    
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses"""
        pass
    
    def describe(self):
        """Concrete method - can be inherited as-is"""
        return f"This shape has area {self.area()} and perimeter {self.perimeter()}"

# Cannot instantiate abstract class
try:
    shape = Shape()
except TypeError as e:
    print(e)  # Can't instantiate abstract class Shape with abstract methods area, perimeter

# Concrete implementation
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# Now we can instantiate
rect = Rectangle(5, 10)
print(rect.area())        # 50
print(rect.perimeter())   # 30
print(rect.describe())    # This shape has area 50 and perimeter 30

# Incomplete implementation raises error
class IncompleteShape(Shape):
    def area(self):
        return 0
    # Missing perimeter() implementation!

try:
    incomplete = IncompleteShape()
except TypeError as e:
    print(e)  # Can't instantiate abstract class IncompleteShape with abstract methods perimeter
```

#### Abstract Properties

```py
from abc import ABC, abstractmethod

class Vehicle(ABC):
    """Abstract vehicle with abstract properties"""
    
    @property
    @abstractmethod
    def max_speed(self):
        """Maximum speed - must be defined in subclasses"""
        pass
    
    @property
    @abstractmethod
    def capacity(self):
        """Passenger capacity"""
        pass
    
    @abstractmethod
    def start_engine(self):
        """Abstract method"""
        pass

class Car(Vehicle):
    def __init__(self, model, speed, seats):
        self.model = model
        self._max_speed = speed
        self._capacity = seats
    
    @property
    def max_speed(self):
        return self._max_speed
```

#### Abstract Class Methods and Static Methods

```py
from abc import ABC, abstractmethod

class DataParser(ABC):
    """Abstract parser for different data formats"""
    
    @abstractmethod
    def parse(self, data):
        """Instance method - parse data"""
        pass
    
    @classmethod
    @abstractmethod
    def from_file(cls, filename):
        """Abstract class method - create parser from file"""
        pass
    
    @staticmethod
    @abstractmethod
    def validate_format(data):
        """Abstract static method - validate data format"""
        pass

class JSONParser(DataParser):
    def __init__(self, data):
        self.data = data
    
    def parse(self):
        import json
        return json.loads(self.data)
    
    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            return cls(f.read())
    
    @staticmethod
    def validate_format(data):
        import json
        try:
            json.loads(data)
            return True
        except json.JSONDecodeError:
            return False

# Using the concrete implementation
parser = JSONParser('{"key": "value"}')
print(parser.parse())  # {'key': 'value'}
print(JSONParser.validate_format('{"valid": true}'))  # True
print(JSONParser.validate_format('invalid'))  # False
```

#### Duck Typing vs ABCs

Python’s dynamic typing style — “If it walks like a duck and quacks like a duck, it’s a duck.”

```py
from abc import ABC, abstractmethod

class Readable(ABC):
    """Explicit interface definition"""
    
    @abstractmethod
    def read(self):
        pass

class FileReader(Readable):
    def __init__(self, filename):
        self.filename = filename
    
    def read(self):
        with open(self.filename, 'r') as f:
            return f.read()

class DatabaseReader(Readable):
    def __init__(self, query):
        self.query = query
    
    def read(self):
        # Simulate database read
        return f"Data from query: {self.query}"

def process_readable(readable: Readable):
    """Type hint indicates expected interface"""
    if not isinstance(readable, Readable):
        raise TypeError("Object must implement Readable interface")
    return readable.read()

# Both work
file_reader = FileReader('test.txt')
db_reader = DatabaseReader('SELECT * FROM users')

print(process_readable(file_reader))
print(process_readable(db_reader))
```

- **Duck typing approach**

```py
# Duck typing approach (Pythonic)
def process_file(file_obj):
    """Accepts any object with read() method"""
    content = file_obj.read()
    return content.upper()

# Works with file
with open('test.txt', 'w') as f:
    f.write('hello world')

with open('test.txt', 'r') as f:
    print(process_file(f))

# Works with StringIO (duck typing)
from io import StringIO
string_file = StringIO('hello from stringio')
print(process_file(string_file))

# Works with any custom class with read()
class CustomReader:
    def read(self):
        return "custom content"

print(process_file(CustomReader()))
```

- When to use each:
    - Duck Typing is the default or pythonic way for flexibility
    - Use ABCs when you need explicit contracts, framework development, or type safety

#### Virtual Subclasses

ABCs can register classes as "***virtual subclasses***" without actual inheritance.

```py
from abc import ABC, abstractmethod

class Sized(ABC):
    """ABC for objects with size"""
    
    @abstractmethod
    def __len__(self):
        pass

class MyList:
    """Custom list without inheriting from Sized"""
    
    def __init__(self, items):
        self.items = items
    
    def __len__(self):
        return len(self.items)

# Register as virtual subclass
Sized.register(MyList)

my_list = MyList([1, 2, 3])

# isinstance check passes!
print(isinstance(my_list, Sized))  # True
print(issubclass(MyList, Sized))   # True

# But not in __mro__
print(Sized in MyList.__mro__)     # False
```

- Alternative: Use `__subclasshook__` for automatic virtual subclass detection:

```py
from abc import ABC, abstractmethod

class Sized(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        """Automatically recognize classes with __len__"""
        if cls is Sized:
            if any("__len__" in B.__dict__ for B in subclass.__mro__):
                return True
        return NotImplemented

class MyCollection:
    """Has __len__ but doesn't inherit Sized"""
    
    def __init__(self):
        self.items = []
    
    def __len__(self):
        return len(self.items)

# Automatically recognized as Sized!
print(isinstance(MyCollection(), Sized))  # True
print(issubclass(MyCollection, Sized))    # True
```

#### Built-in ABCs

Python's `collections.abc` module provides many useful ABCs:

```py
from collections.abc import Iterable, Sequence

# Custom iterable using ABC
class CountDown(Iterable):
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

countdown = CountDown(5)
print(isinstance(countdown, Iterable))  # True
print(list(countdown))  # [5, 4, 3, 2, 1]

# Custom sequence
class ImmutableList(Sequence):
    """Immutable sequence implementation"""
    
    def __init__(self, items):
        self._items = tuple(items)
    
    def __getitem__(self, index):
        return self._items[index]
    
    def __len__(self):
        return len(self._items)

immutable = ImmutableList([1, 2, 3, 4, 5])
print(immutable[2])         # 3
print(len(immutable))       # 5
print(3 in immutable)       # True (inherited from Sequence)
print(immutable.index(4))   # 3 (inherited method)
print(immutable.count(2))   # 1 (inherited method)
```

| ABC | Required Methods | Inherited Methods |
| :--- | :--- | :--- |
| **Iterable** | `__iter__` | - |
| **Iterator** | `__iter__`, `__next__` | - |
| **Sized** | `__len__` | - |
| **Container** | `__contains__` | - |
| **Sequence** | `__getitem__`, `__len__` | `__contains__`, `__iter__`, `__reversed__`, `index`, `count` |
| **MutableSequence** | `__getitem__`, `__setitem__`, `__delitem__`, `__len__`, `insert` | All Sequence methods plus `append`, `reverse`, `extend`, `pop`, `remove`, `__iadd__` |
| **Mapping** | `__getitem__`, `__iter__`, `__len__` | `__contains__`, `keys`, `items`, `values`, `get`, `__eq__`, `__ne__` |
| **MutableMapping** | `__getitem__`, `__setitem__`, `__delitem__`, `__iter__`, `__len__` | All Mapping methods plus `pop`, `popitem`, `clear`, `update`, `setdefault` |


- **Practical Example**: Plugin System with ABCs

```py
from abc import ABC, abstractmethod
from typing import Dict, Any

class Plugin(ABC):
    """Abstract base class for plugins"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize plugin with configuration"""
        pass
    
    @abstractmethod
    def execute(self, data: Any) -> Any:
        """Execute plugin functionality"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        pass

class LoggingPlugin(Plugin):
    """Concrete logging plugin"""
    
    @property
    def name(self):
        return "Logger"
    
    @property
    def version(self):
        return "1.0.0"
    
    def initialize(self, config):
        self.log_file = config.get('log_file', 'app.log')
        self.logs = []
    
    def execute(self, data):
        log_entry = f"[{self.name}] {data}"
        self.logs.append(log_entry)
        return log_entry
    
    def cleanup(self):
        # Write logs to file
        print(f"Writing {len(self.logs)} logs to {self.log_file}")

class DataValidationPlugin(Plugin):
    """Concrete validation plugin"""
    
    @property
    def name(self):
        return "Validator"
    
    @property
    def version(self):
        return "2.0.0"
    
    def initialize(self, config):
        self.rules = config.get('rules', [])
    
    def execute(self, data):
        # Validate data against rules
        return all(rule(data) for rule in self.rules)
    
    def cleanup(self):
        print(f"Validation plugin cleanup complete")

# Plugin manager
class PluginManager:
    def __init__(self):
        self.plugins = []
    
    def register(self, plugin: Plugin):
        """Only accept Plugin instances"""
        if not isinstance(plugin, Plugin):
            raise TypeError("Must be a Plugin instance")
        self.plugins.append(plugin)
    
    def initialize_all(self, config):
        for plugin in self.plugins:
            plugin.initialize(config.get(plugin.name, {}))
    
    def execute_all(self, data):
        results = {}
        for plugin in self.plugins:
            results[plugin.name] = plugin.execute(data)
        return results
    
    def cleanup_all(self):
        for plugin in self.plugins:
            plugin.cleanup()

# Using the plugin system
manager = PluginManager()
manager.register(LoggingPlugin())
manager.register(DataValidationPlugin())

config = {
    "Logger": {"log_file": "output.log"},
    "Validator": {"rules": [lambda x: len(x) > 0]}
}

manager.initialize_all(config)
results = manager.execute_all("test data")
print(results)
manager.cleanup_all()
```

### Metaclasses and Dynamic Class Creation

In Python, classes are objects created by *metaclasses*. The default metaclass is `type`.

```py
# Classes are instances of type
class MyClass:
    pass

print(type(MyClass))  # <class 'type'>
print(isinstance(MyClass, type))  # True

# Creating a class dynamically with type()
# type(name, bases, dict)
DynamicClass = type('DynamicClass', (), {'x': 10, 'method': lambda self: self.x * 2})

obj = DynamicClass()
print(obj.x)          # 10
print(obj.method())   # 20

# Equivalent to:
class DynamicClass:
    x = 10
    def method(self):
        return self.x * 2

```

#### Creating Custom Metaclasses

Metaclasses allow you to customize class creation behavior.

```py
class SingletonMeta(type):
    """Metaclass that implements Singleton pattern"""
    
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        """Called when creating an instance of the class"""
        if cls not in cls._instances:
            # Create the instance only once
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    """Database class with Singleton pattern"""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"Database initialized with {connection_string}")

# Creating instances
db1 = Database("localhost:5432")  # Database initialized with localhost:5432
db2 = Database("localhost:3306")  # No output - returns existing instance

print(db1 is db2)  # True (same object)
print(db1.connection_string)  # localhost:5432 (original connection)
```

#### Metaclass `__new__ `and `__init__`

```py
class Meta(type):
    """Custom metaclass demonstrating __new__ and __init__"""
    
    def __new__(mcs, name, bases, namespace):
        """Called to create the class object"""
        print(f"Meta.__new__ called for class {name}")
        
        # Modify class before creation
        namespace['created_by_meta'] = True
        
        # Create the class
        cls = super().__new__(mcs, name, bases, namespace)
        return cls
    
    def __init__(cls, name, bases, namespace):
        """Called to initialize the class object"""
        print(f"Meta.__init__ called for class {name}")
        super().__init__(name, bases, namespace)
    
    def __call__(cls, *args, **kwargs):
        """Called when creating an instance"""
        print(f"Meta.__call__ called for class {cls.__name__}")
        instance = super().__call__(*args, **kwargs)
        return instance

class MyClass(metaclass=Meta):
    """Class using custom metaclass"""
    
    def __init__(self, value):
        print(f"MyClass.__init__ called with value={value}")
        self.value = value

# Class creation triggers Meta.__new__ and Meta.__init__
# Output:
# Meta.__new__ called for class MyClass
# Meta.__init__ called for class MyClass

# Instance creation triggers Meta.__call__ and MyClass.__init__
obj = MyClass(42)
# Output:
# Meta.__call__ called for class MyClass
# MyClass.__init__ called with value=42

print(obj.created_by_meta)  # True (added by metaclass)
```

#### Practical Metaclass Example

- **Validation**

```py
class ValidatedMeta(type):
    """Metaclass that enforces type hints at class level"""
    
    def __new__(mcs, name, bases, namespace):
        # Get type hints
        annotations = namespace.get('__annotations__', {})
        
        # Create validation method
        def __init__(self, **kwargs):
            for attr_name, attr_type in annotations.items():
                if attr_name in kwargs:
                    value = kwargs[attr_name]
                    if not isinstance(value, attr_type):
                        raise TypeError(
                            f"{attr_name} must be {attr_type.__name__}, "
                            f"got {type(value).__name__}"
                        )
                    setattr(self, attr_name, value)
                else:
                    raise ValueError(f"Missing required attribute: {attr_name}")
        
        namespace['__init__'] = __init__
        return super().__new__(mcs, name, bases, namespace)

class Person(metaclass=ValidatedMeta):
    """Class with automatic validation"""
    name: str
    age: int
    email: str

# Valid creation
person1 = Person(name="Alice", age=30, email="alice@example.com")
print(person1.name)  # Alice

# Invalid type
try:
    person2 = Person(name="Bob", age="thirty", email="bob@example.com")
except TypeError as e:
    print(e)  # age must be int, got str

# Missing attribute
try:
    person3 = Person(name="Charlie", age=25)
except ValueError as e:
    print(e)  # Missing required attribute: email
```

- **Automatic Registration**

```py
class RegistryMeta(type):
    """Metaclass that automatically registers classes"""
    
    registry = {}
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        
        # Don't register the base class
        if bases:
            mcs.registry[name] = cls
        
        return cls
    
    @classmethod
    def get_registry(mcs):
        return mcs.registry.copy()

class Command(metaclass=RegistryMeta):
    """Base command class"""
    pass

class CreateCommand(Command):
    """Create command"""
    def execute(self):
        return "Creating..."

class DeleteCommand(Command):
    """Delete command"""
    def execute(self):
        return "Deleting..."

class UpdateCommand(Command):
    """Update command"""
    def execute(self):
        return "Updating..."

# All commands automatically registered
print(RegistryMeta.get_registry())
# {'CreateCommand': <class 'CreateCommand'>, 
#  'DeleteCommand': <class 'DeleteCommand'>, 
#  'UpdateCommand': <class 'UpdateCommand'>}

# Dynamic command execution
command_name = "CreateCommand"
command_class = RegistryMeta.registry[command_name]
command = command_class()
print(command.execute())  # Creating...
```

#### Metaclass vs Class Decorators

Often class decorators can achieve similar results with simpler syntax:

```py
# Using metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DatabaseMeta(metaclass=SingletonMeta):
    pass

# Using decorator (simpler!)
def singleton(cls):
    """Singleton decorator"""
    instances = {}
    
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class DatabaseDecorator:
    pass

# Both work identically
db1 = DatabaseDecorator()
db2 = DatabaseDecorator()
print(db1 is db2)  # True
```

- When to use:
    - **Metaclass**: When you need to modify class creation itself, affect inheritance, or work with class-level operations
    - **Decorator**: When you need to modify/wrap class behavior (simpler and more readable)

- Most problems can be solved without metaclasses hence avoid or use class decorator when possible
- Metaclasses are "magic" and hard to understand and Metaclass conflicts can be complex
- Frameworks like **Django** **ORM** and **SQLAlchemy** use metaclasses to auto-register models and manage fields dynamically.

[OOPS Notebook](./Notebooks/9_Object_Oriented_Programming.ipynb)

---

## Functional Programming in Python

**Functional Programming (FP)** is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing state and mutable data.

### FP vs OOP

| Aspect | Object-Oriented Programming | Functional Programming |
| :--- | :--- | :--- |
| **Data & Functions** | Combined in objects | Separated (data and functions are independent) |
| **State** | Mutable state in objects | Immutable data structures |
| **Primary Focus** | Objects and their interactions | Functions and their composition |
| **Side Effects** | Common (modifying object state) | Avoided (pure functions) |
| **Flow Control** | Loops, conditionals | Recursion, higher-order functions |
| **Main Benefit** | Encapsulation, modeling real-world entities | Predictability, testability, parallelization |

### Core Principles of Functional Programming

| Concept | Description |
|----------|--------------|
| **Pure Functions** | Functions that always produce the same output for the same input and have no side effects. |
| **Immutability** | Data is never modified; instead, new data structures are created. |
| **First-Class Functions** | Functions are treated as objects; they can be passed, returned, and stored in variables. |
| **Higher-Order Functions** | Functions that take or return other functions. |
| **Function Composition** | Building complex operations from simple functions |
| **Lazy Evaluation** | Delay computation until the result is needed. |
| **Declarative Style** | Describing "What to do" and not "How to do it" |

```py
# OOP Approach
class ShoppingCart:
    def __init__(self):
        self.items = []
        self.total = 0
    
    def add_item(self, item, price):
        self.items.append(item)
        self.total += price

cart = ShoppingCart()
cart.add_item("Book", 20)
cart.add_item("Pen", 5)

# FP Approach
def add_item(cart, item, price):
    """Returns a new cart with added item"""
    return {
        'items': cart['items'] + [item],
        'total': cart['total'] + price
    }

cart = {'items': [], 'total': 0}
cart = add_item(cart, "Book", 20)
cart = add_item(cart, "Pen", 5)
```

#### 1. Pure Functions

A **pure function**:

- Always returns the same output for the same input (**deterministic**)
- Has no side effects (doesn't modify external state)
- Doesn't depend on external mutable state

```py
# ✅ Pure Function
def calculate_tax(amount, rate):
    """Pure: Same inputs always produce same output, no side effects"""
    return amount * rate

print(calculate_tax(100, 0.1))  # Always 10.0
print(calculate_tax(100, 0.1))  # Always 10.0

# ❌ Impure Function - Modifies global state
total = 0

def add_to_total(amount):
    """Impure: Modifies external state"""
    global total
    total += amount  # Side effect!
    return total

print(add_to_total(10))  # 10
print(add_to_total(10))  # 20 (different output for same input!)

# ❌ Impure Function - Depends on external state
tax_rate = 0.1

def calculate_tax_impure(amount):
    """Impure: Depends on external mutable state"""
    return amount * tax_rate  # Depends on global variable

print(calculate_tax_impure(100))  # 10.0
tax_rate = 0.2
print(calculate_tax_impure(100))  # 20.0 (same input, different output!)

# ❌ Impure Function - I/O operations
def log_and_calculate(amount, rate):
    """Impure: Has I/O side effects"""
    print(f"Calculating tax for {amount}")  # Side effect (I/O)
    return amount * rate
```

**Benefits of Pure Functions**:

- **Testability**: Easy to test (no setup/teardown needed)
- **Predictability**: Same input = same output
- **Parallelization**: Safe to run concurrently
- **Caching/Memoization**: Results can be cached
- **Debugging**: Easier to reason about

Real world example of a pure function

```py
# Data validation (pure)
import re
from typing import List, Dict

def validate_email(email: str) -> bool:
    """Pure function for email validation"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_user_data(user: Dict) -> Dict:
    """Pure function returning validation results"""
    return {
        'email_valid': validate_email(user.get('email', '')),
        'age_valid': isinstance(user.get('age'), int) and 0 < user.get('age', 0) < 150,
        'name_valid': isinstance(user.get('name'), str) and len(user.get('name', '')) > 0
    }

# Usage
user = {'email': 'test@example.com', 'age': 25, 'name': 'John'}
print(validate_user_data(user))
# {'email_valid': True, 'age_valid': True, 'name_valid': True}

# Always same result for same input
print(validate_user_data(user))
# {'email_valid': True, 'age_valid': True, 'name_valid': True}
```

#### 2. Immutability

Immutability means data cannot be changed after creation. Instead of modifying existing data, create new data with desired changes.

```py
# ❌ Mutable approach (avoiding in FP)
def add_item_mutable(cart, item):
    """Modifies original list"""
    cart.append(item)
    return cart

cart = ['apple']
new_cart = add_item_mutable(cart, 'banana')
print(cart)      # ['apple', 'banana'] - Original modified!
print(new_cart)  # ['apple', 'banana'] - Same object

# ✅ Immutable approach (FP style)
def add_item_immutable(cart, item):
    """Returns new list without modifying original"""
    return cart + [item]  # Creates new list

cart = ['apple']
new_cart = add_item_immutable(cart, 'banana')
print(cart)      # ['apple'] - Original unchanged
print(new_cart)  # ['apple', 'banana'] - New list
```

- **Working with Immutable Data Structures**

```py
from typing import List, NamedTuple, Tuple
from dataclasses import dataclass

# Using NamedTuple (immutable)
class CartItem(NamedTuple):
    product_id: str
    name: str
    price: float
    quantity: int

def update_quantity(cart: List[CartItem], product_id: str, new_quantity: int) -> List[CartItem]:
    """Returns new cart with updated quantity"""
    return [
        item._replace(quantity=new_quantity) if item.product_id == product_id else item
        for item in cart
    ]

# Using frozen dataclass (Python 3.7+)
@dataclass(frozen=True)
class Product:
    id: str
    name: str
    price: float

# product.price = 100  # FrozenInstanceError

# Usage example
cart = [
    CartItem("001", "Laptop", 999.99, 1),
    CartItem("002", "Mouse", 29.99, 2)
]

new_cart = update_quantity(cart, "002", 3)
print(f"Original: {cart[1].quantity}")  # 2
print(f"Updated: {new_cart[1].quantity}")  # 3
```

- **Immutable Operations on Common Data Structures**

```py
# List operations (immutable style)
original = [1, 2, 3]

# Append
new_list = original + [4]           # [1, 2, 3, 4]

# Remove
new_list = [x for x in original if x != 2]  # [1, 3]

# Update at index
new_list = original[:1] + [99] + original[2:]  # [1, 99, 3]

# Dictionary operations (immutable style)
original_dict = {'a': 1, 'b': 2}

# Add/update
new_dict = {**original_dict, 'c': 3}  # {'a': 1, 'b': 2, 'c': 3}

# Remove
new_dict = {k: v for k, v in original_dict.items() if k != 'a'}  # {'b': 2}

# Update value
new_dict = {**original_dict, 'a': 99}  # {'a': 99, 'b': 2}

# Using frozenset (immutable set)
immutable_set = frozenset([1, 2, 3])
# immutable_set.add(4)  # AttributeError
new_set = immutable_set | {4}  # frozenset({1, 2, 3, 4})
```

> A function is **referentially transparent** if it can be replaced with its return value without changing program behavior.

#### First class Functions

In Python, functions are first-class citizens, meaning they can be:

- Assigned to variables
- Passed as arguments to other functions
- Returned from other functions
- Stored in data structures

```py
# Functions as variables
def greet(name):
    return f"Hello, {name}!"

say_hello = greet  # Assign function to variable
print(say_hello("Alice"))  # Hello, Alice!

# Functions in data structures
operations = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y,
    'divide': lambda x, y: x / y if y != 0 else None
}

print(operations['add'](5, 3))       # 8
print(operations['multiply'](4, 7))  # 28

# List of functions
validators = [
    lambda x: x > 0,           # Check positive
    lambda x: x % 2 == 0,      # Check even
    lambda x: x < 100          # Check less than 100
]

def validate_all(value, validators):
    return all(validator(value) for validator in validators)

print(validate_all(42, validators))   # True
print(validate_all(-5, validators))   # False
```

#### Higher Order Functions

Higher-order functions are functions that:

- Take one or more functions as arguments OR Return a function as result

```py
# HOF: Taking function as argument
def apply_operation(x, y, operation):
    """Apply operation function to x and y"""
    return operation(x, y)

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

print(apply_operation(5, 3, add))       # 8
print(apply_operation(5, 3, multiply))  # 15

# HOF: Returning function
def make_multiplier(factor):
    """Returns a function that multiplies by factor"""
    def multiplier(x):
        return x * factor
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15

# Practical example: Logger factory
def create_logger(prefix):
    """Returns a logging function with specific prefix"""
    def logger(message):
        print(f"[{prefix}] {message}")
    return logger

error_logger = create_logger("ERROR")
info_logger = create_logger("INFO")

error_logger("Something went wrong")  # [ERROR] Something went wrong
info_logger("Process completed")      # [INFO] Process completed
```

#### Function Composition

Combining two or more functions to create a new function where the output of one function becomes the input of the next.

- We can manually combine functions or use a `compose` helper function

```py
def compose(*functions):
    """Compose multiple functions: compose(f, g, h)(x) = f(g(h(x)))"""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner

# Example functions
def add_one(x):
    return x + 1

def double(x):
    return x * 2

def square(x):
    return x ** 2

# Compose: square(double(add_one(x)))
composed = compose(square, double, add_one)
print(composed(3))  # square(double(add_one(3))) = square(double(4)) = square(8) = 64

# Alternative: Using reduce
from functools import reduce

def compose_with_reduce(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

composed2 = compose_with_reduce(square, double, add_one)
print(composed2(3))  # 64
```

#### Partial Application

Partial application is a technique where a function is applied to some, but not all, of its arguments, resulting in a new function that takes the remaining arguments. This new function is a specialized version of the original function, with some of its parameters pre-filled or "fixed."

```py
from functools import partial

def power(base, exponent):
    return base ** exponent

# Create specialized functions
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125
```

```py
# Practical example: API client with default headers
import requests

def make_request(url, headers=None, timeout=30):
    return requests.get(url, headers=headers, timeout=timeout)

# Create API-specific request function
api_request = partial(
    make_request,
    headers={'Authorization': 'Bearer token123'},
    timeout=10
)

# Now use with just URL
# response = api_request('https://api.example.com/data')
```

### Built-in Functional Tools

#### `map()`

Applies a function to every item in an **iterable**, returning an **iterator**.

```py
# Syntax: map(function, iterable, ...)

# Basic usage
def square(x):
    return x ** 2

numbers = [1, 2, 3, 4, 5]
squared = map(square, numbers)
print(list(squared))  # [1, 4, 9, 16, 25]

# With lambda
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x ** 2, numbers)
print(list(squared))  # [1, 4, 9, 16, 25]

# Multiple iterables
def add(x, y):
    return x + y

list1 = [1, 2, 3]
list2 = [10, 20, 30]
result = map(add, list1, list2)
print(list(result))  # [11, 22, 33]
```

```py
# Real-world example: Data transformation
users = [
    {'name': 'Alice', 'age': 30},
    {'name': 'Bob', 'age': 25},
    {'name': 'Charlie', 'age': 35}
]

# Extract names
names = list(map(lambda user: user['name'], users))
print(names)  # ['Alice', 'Bob', 'Charlie']

# Transform data
def format_user(user):
    return f"{user['name']} ({user['age']} years old)"

formatted = list(map(format_user, users))
print(formatted)
# ['Alice (30 years old)', 'Bob (25 years old)', 'Charlie (35 years old)']

# map() returns an iterator (lazy evaluation)
squared = map(lambda x: x ** 2, range(1000000))
print(squared)  # <map object> - not computed yet
# Values computed only when iterated
```

- For some tasks both `map()` and list comprehension can produce same result. List comprehension is more Pythonic and often more readable. Use `map()` when:
    - You have an existing named function
    - Working with multiple iterables
    - Prefer lazy evaluation

#### `filter()`

Filters items from an iterable based on a function that returns True/False.

```py
# Syntax: filter(function, iterable)

# Basic usage
def is_even(x):
    return x % 2 == 0

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = filter(is_even, numbers)
print(list(even_numbers))  # [2, 4, 6, 8, 10]

# With lambda
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
even_numbers = filter(lambda x: x % 2 == 0, numbers)
print(list(even_numbers))  # [2, 4, 6, 8, 10]

# Filter with None (removes falsy values)
mixed = [0, 1, False, True, '', 'hello', None, [], [1, 2]]
truthy = filter(None, mixed)
print(list(truthy))  # [1, True, 'hello', [1, 2]]
```

```py
# Real-world example: Data filtering
users = [
    {'name': 'Alice', 'age': 30, 'active': True},
    {'name': 'Bob', 'age': 17, 'active': True},
    {'name': 'Charlie', 'age': 25, 'active': False},
    {'name': 'David', 'age': 40, 'active': True}
]

# Filter active adult users
def is_active_adult(user):
    return user['active'] and user['age'] >= 18

active_adults = list(filter(is_active_adult, users))
print(active_adults)
# [{'name': 'Alice', 'age': 30, 'active': True}, 
#  {'name': 'David', 'age': 40, 'active': True}]

# Chaining filter and map
names = list(map(
    lambda user: user['name'],
    filter(is_active_adult, users)
))
print(names)  # ['Alice', 'David']
```

#### `reduce()`

Applies a function cumulatively to items in an iterable, reducing it to a single value.

```py
from functools import reduce

# Syntax: reduce(function, iterable[, initial])

# Basic usage: Sum of numbers
def add(x, y):
    return x + y

numbers = [1, 2, 3, 4, 5]
total = reduce(add, numbers)
print(total)  # 15

# How it works step by step:
# add(1, 2) → 3
# add(3, 3) → 6
# add(6, 4) → 10
# add(10, 5) → 15

# With lambda
numbers = [1, 2, 3, 4, 5]
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120 (1 * 2 * 3 * 4 * 5)

# With initial value
numbers = [1, 2, 3, 4, 5]
total = reduce(lambda x, y: x + y, numbers, 10)
print(total)  # 25 (10 + 1 + 2 + 3 + 4 + 5)
```

```py
# Real-world examples

# 1. Find maximum
numbers = [3, 7, 2, 9, 1, 5]
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 9

# 2. Flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda x, y: x + y, nested)
print(flattened)  # [1, 2, 3, 4, 5, 6]

# 3. Count occurrences
words = ['apple', 'banana', 'apple', 'cherry', 'banana', 'apple']
word_count = reduce(
    lambda acc, word: {**acc, word: acc.get(word, 0) + 1},
    words,
    {}
)
print(word_count)  # {'apple': 3, 'banana': 2, 'cherry': 1}

# 4. Compose functions
functions = [
    lambda x: x + 1,
    lambda x: x * 2,
    lambda x: x ** 2
]

composed = reduce(lambda f, g: lambda x: g(f(x)), functions)
print(composed(3))  # ((3 + 1) * 2) ** 2 = (4 * 2) ** 2 = 8 ** 2 = 64
```

- Use `reduce()` when its necessary, for simpler tasks, always use buit-ins (sum of an iterable)
- For complex accumulations use `reduce()`

```py
# Example: Group by category
products = [
    {'name': 'Apple', 'category': 'Fruit'},
    {'name': 'Carrot', 'category': 'Vegetable'},
    {'name': 'Banana', 'category': 'Fruit'},
    {'name': 'Broccoli', 'category': 'Vegetable'}
]

grouped = reduce(
    lambda acc, product: {
        **acc,
        product['category']: acc.get(product['category'], []) + [product['name']]
    },
    products,
    {}
)
print(grouped)
# {'Fruit': ['Apple', 'Banana'], 'Vegetable': ['Carrot', 'Broccoli']}
```

#### `zip()`

Combines multiple iterables element-wise into tuples.

```py
# Syntax: zip(*iterables)

# Basic usage
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 35]

combined = zip(names, ages)
print(list(combined))  # [('Alice', 30), ('Bob', 25), ('Charlie', 35)]

# Multiple iterables
names = ['Alice', 'Bob']
ages = [30, 25]
cities = ['New York', 'London']

combined = zip(names, ages, cities)
print(list(combined))  # [('Alice', 30, 'New York'), ('Bob', 25, 'London')]

# Stops at shortest iterable
list1 = [1, 2, 3]
list2 = ['a', 'b']  # Shorter

result = zip(list1, list2)
print(list(result))  # [(1, 'a'), (2, 'b')] - Stops at 2

# zip with strict=True (Python 3.10+) - raises error if lengths differ
# result = zip(list1, list2, strict=True)  # ValueError

# Unzipping
pairs = [('Alice', 30), ('Bob', 25), ('Charlie', 35)]
names, ages = zip(*pairs)  # Unpack with *
print(names)  # ('Alice', 'Bob', 'Charlie')
print(ages)   # (30, 25, 35)
```

```py
# Real-world examples

# 1. Create dictionary from two lists
keys = ['name', 'age', 'city']
values = ['Alice', 30, 'New York']
person = dict(zip(keys, values))
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'New York'}

# 2. Parallel iteration
questions = ['Name?', 'Age?', 'City?']
answers = ['Alice', '30', 'NYC']

for question, answer in zip(questions, answers):
    print(f"{question} {answer}")
# Name? Alice
# Age? 30
# City? NYC

# 3. Matrix transposition
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

transposed = list(zip(*matrix))
print(transposed)  # [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

# Convert to list of lists
transposed = [list(row) for row in zip(*matrix)]
print(transposed)  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# 4. Pairwise iteration
numbers = [1, 2, 3, 4, 5]
pairs = list(zip(numbers, numbers[1:]))
print(pairs)  # [(1, 2), (2, 3), (3, 4), (4, 5)]
```

#### `enumerate()`

Returns an iterator of tuples containing indices and values.

```py
# Syntax: enumerate(iterable, start=0)

# Basic usage
fruits = ['apple', 'banana', 'cherry']

for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Custom start index
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}. {fruit}")
# 1. apple
# 2. banana
# 3. cherry

# Get list of tuples
fruits = ['apple', 'banana', 'cherry']
indexed = list(enumerate(fruits))
print(indexed)  # [(0, 'apple'), (1, 'banana'), (2, 'cherry')]
```

```py
# Real-world examples

# 1. Finding indices of matching elements
numbers = [10, 20, 30, 20, 40]
indices_of_20 = [i for i, x in enumerate(numbers) if x == 20]
print(indices_of_20)  # [1, 3]

# 2. Enumerate with conditional
words = ['apple', 'banana', 'apricot', 'blueberry']
a_words = {i: word for i, word in enumerate(words) if word.startswith('a')}
print(a_words)  # {0: 'apple', 2: 'apricot'}

# 3. Processing with context
lines = ['First line', 'Second line', 'Third line']
for i, line in enumerate(lines, start=1):
    print(f"Line {i}: {line}")
# Line 1: First line
# Line 2: Second line
# Line 3: Third line

# 4. Tracking position in nested structures
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
for i, row in enumerate(matrix):
    for j, value in enumerate(row):
        print(f"matrix[{i}][{j}] = {value}")
```

#### Lambda Functions

Lambda functions are small anonymous functions defined with the lambda keyword.

```py
# Syntax: lambda arguments: expression

# Basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# No arguments
get_pi = lambda: 3.14159
print(get_pi())  # 3.14159

# Lambda with conditional
max_of_two = lambda x, y: x if x > y else y
print(max_of_two(10, 20))  # 20

# Common use cases

# 1. With sorted()
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]

# Sort by grade
sorted_students = sorted(students, key=lambda student: student['grade'])
print(sorted_students)
# [{'name': 'Charlie', 'grade': 78}, {'name': 'Alice', 'grade': 85}, ...]

# 2. With map()
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# 3. With filter()
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# 4. Immediate execution (IIFE - Immediately Invoked Function Expression)
result = (lambda x, y: x + y)(5, 3)
print(result)  # 8
```

- Limitations of lambda functions
    - Cannot contain statements
    - Cannot have multiple expressions
    - Cannot have annotations
    - Difficult to debug (no name in traceback)

- When to use lambda functions:
    - Simple, one-line operations
    - Throwaway functions
    - As argument to higher-order functions

### Closures

A closure is a function that remembers values from its enclosing scope, even after that scope has finished executing.

```py
def outer_function(message):
    """Outer function that defines closure"""
    # message is in the enclosing scope
    
    def inner_function():
        """Inner function - closure"""
        print(message)  # Accesses variable from outer scope
    
    return inner_function

# Create closure
my_func = outer_function("Hello, World!")
my_func()  # Hello, World!

# The outer function has finished, but inner still has access to 'message'
another_func = outer_function("Goodbye!")
another_func()  # Goodbye!
```

- **How closures work**

```py
def make_counter():
    """Factory function that creates counter closures"""
    count = 0  # Free variable (captured by closure)
    
    def increment():
        nonlocal count  # Modify variable from enclosing scope
        count += 1
        return count
    
    return increment

# Create independent counters
counter1 = make_counter()
counter2 = make_counter()

print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

print(counter2())  # 1 (independent from counter1)
print(counter2())  # 2

# Inspecting closures
print(counter1.__closure__)  # (<cell at 0x...: int object at 0x...>,)
print(counter1.__closure__[0].cell_contents)  # 3 (current count value)
```

#### Practical example of closure

```py
# 1. Configuration/Settings
def create_multiplier(factor):
    """Create specialized multiplication function"""
    def multiply(x):
        return x * factor
    return multiply

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# 2. Private state (data hiding)
def create_account(initial_balance):
    """Create account with private balance"""
    balance = initial_balance  # Private variable
    
    def deposit(amount):
        nonlocal balance
        if amount > 0:
            balance += amount
            return f"Deposited ${amount}. New balance: ${balance}"
        return "Invalid amount"
    
    def withdraw(amount):
        nonlocal balance
        if 0 < amount <= balance:
            balance -= amount
            return f"Withdrew ${amount}. New balance: ${balance}"
        return "Insufficient funds or invalid amount"
    
    def get_balance():
        return balance
    
    return {
        'deposit': deposit,
        'withdraw': withdraw,
        'balance': get_balance
    }

# Usage
account = create_account(1000)
print(account['deposit'](500))    # Deposited $500. New balance: $1500
print(account['withdraw'](200))   # Withdrew $200. New balance: $1300
print(account['balance']())       # 1300
# No direct access to balance variable!

# 3. Memoization (caching)
def memoize(func):
    """Decorator using closure for caching"""
    cache = {}  # Captured by closure
    
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    return wrapper

@memoize
def fibonacci(n):
    """Fibonacci with memoization"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Fast due to caching!

# 4. Callback with context
def create_button_handler(button_id):
    """Create event handler with captured context"""
    def handler():
        print(f"Button {button_id} clicked!")
    return handler

button1_handler = create_button_handler(1)
button2_handler = create_button_handler(2)

button1_handler()  # Button 1 clicked!
button2_handler()  # Button 2 clicked!

# 5. Function factory with configuration
def create_validator(min_value, max_value):
    """Create validator with specific range"""
    def validate(value):
        return min_value <= value <= max_value
    return validate

age_validator = create_validator(0, 120)
percentage_validator = create_validator(0, 100)

print(age_validator(25))    # True
print(age_validator(150))   # False
print(percentage_validator(95))  # True
```

#### Closure Pitfalls

- Late binding in loops

```py
def create_multipliers_wrong():
    """WRONG: All functions will multiply by 3"""
    multipliers = []
    for i in range(1, 4):
        multipliers.append(lambda x: x * i)
    return multipliers

funcs = create_multipliers_wrong()
print(funcs[0](10))  # Expected 10, got 30!
print(funcs[1](10))  # Expected 20, got 30!
print(funcs[2](10))  # Expected 30, got 30!

# Fix 1: Use default argument
def create_multipliers_fix1():
    """CORRECT: Capture i with default argument"""
    multipliers = []
    for i in range(1, 4):
        multipliers.append(lambda x, i=i: x * i)  # i=i captures current value
    return multipliers

funcs = create_multipliers_fix1()
print(funcs[0](10))  # 10 ✓
print(funcs[1](10))  # 20 ✓
print(funcs[2](10))  # 30 ✓

# Fix 2: Use function factory
def create_multipliers_fix2():
    """CORRECT: Use factory function"""
    def make_multiplier(factor):
        return lambda x: x * factor
    
    return [make_multiplier(i) for i in range(1, 4)]

funcs = create_multipliers_fix2()
print(funcs[0](10))  # 10 ✓
print(funcs[1](10))  # 20 ✓
print(funcs[2](10))  # 30 ✓
```

- Unintended retention of large objects

```py
def create_processor():
    """Memory leak: large_data retained unnecessarily"""
    large_data = [x for x in range(1000000)]  # Large list
    
    def process(index):
        return large_data[index] * 2  # Keeps entire large_data in memory
    
    return process

# Fix: Extract only needed data
def create_processor_efficient():
    """Better: Don't capture unnecessary data"""
    def process(data, index):
        return data[index] * 2
    
    return process
```

- When to use closure
    - Simple state management
    - Single function behaviour
    - Callback function

### Decorators

Decorators are functions that modify the behavior of other functions or classes. They use the `@decorator` syntax.

#### Function Decorators

```py
# Basic decorator structure
def my_decorator(func):
    """Wrapper that adds behavior to func"""
    def wrapper(*args, **kwargs):
        # Before function execution
        print("Before function call")
        
        # Call original function
        result = func(*args, **kwargs)
        
        # After function execution
        print("After function call")
        
        return result
    
    return wrapper

# Using decorator
@my_decorator
def say_hello(name):
    print(f"Hello, {name}!")
    return name

# Equivalent to: say_hello = my_decorator(say_hello)

say_hello("Alice")
# Output:
# Before function call
# Hello, Alice!
# After function call
```

#### Common Decorator Patterns

```py
# 1. Logging decorator
import functools
from datetime import datetime

def log_calls(func):
    """Log function calls with arguments and return value"""
    @functools.wraps(func)  # Preserves original function metadata
    def wrapper(*args, **kwargs):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Calling {func.__name__}")
        print(f"  Args: {args}, Kwargs: {kwargs}")
        
        result = func(*args, **kwargs)
        
        print(f"  Result: {result}")
        return result
    
    return wrapper

@log_calls
def add(x, y):
    return x + y

print(add(5, 3))
# [2024-01-15 10:30:45] Calling add
#   Args: (5, 3), Kwargs: {}
#   Result: 8
# 8
```

```py
# 2. Timer decorator
import time

def timer(func):
    """Measure execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()  # slow_function took 1.0012 seconds
```

```py
# 3. Validation decorator
def validate_positive(func):
    """Ensure all arguments are positive numbers"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, (int, float)) or arg <= 0:
                raise ValueError(f"All arguments must be positive numbers")
        
        return func(*args, **kwargs)
    
    return wrapper

@validate_positive
def calculate_area(width, height):
    return width * height

print(calculate_area(5, 10))  # 50

try:
    calculate_area(-5, 10)  # ValueError
except ValueError as e:
    print(e)
```

```py
# 4. Retry decorator
def retry(max_attempts=3, delay=1):
    """Retry function on exception"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts >= max_attempts:
                        raise
                    print(f"Attempt {attempts} failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
        
        return wrapper
    
    return decorator

@retry(max_attempts=3, delay=0.5)
def unstable_function():
    import random
    if random.random() < 0.7:
        raise ConnectionError("Connection failed")
    return "Success"

# Will retry up to 3 times
# result = unstable_function()
```

```py
# 5. Caching/Memoization decorator
def memoize(func):
    """Cache function results"""
    cache = {}
    
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    
    # Add cache inspection
    wrapper.cache = cache
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(35))  # Fast due to caching
print(fibonacci.cache)  # View cache contents
```

#### Decorators with arguments

```py
def repeat(times):
    """Decorator factory: creates decorator with custom repetitions"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            results = []
            for _ in range(times):
                results.append(func(*args, **kwargs))
            return results
        
        return wrapper
    
    return decorator

@repeat(times=3)
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
# ['Hello, Alice!', 'Hello, Alice!', 'Hello, Alice!']
```

- More complex example: **Rate limiting**

```py
import time
from collections import deque

def rate_limit(calls_per_second):
    """Limit function calls per second"""
    min_interval = 1.0 / calls_per_second
    
    def decorator(func):
        last_called = [0.0]  # Mutable to modify in closure
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            
            last_called[0] = time.time()
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

@rate_limit(calls_per_second=2)  # Max 2 calls per second
def api_call():
    print(f"API called at {time.time()}")

# Will be rate-limited
# for _ in range(5):
#     api_call()
```

#### Class Decorator

```py
# Decorator that modifies class
def add_repr(cls):
    """Add __repr__ method to class"""
    def __repr__(self):
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{cls.__name__}({attrs})"
    
    cls.__repr__ = __repr__
    return cls

@add_repr
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)
print(person)  # Person(name='Alice', age=30)
```

```py
# Singleton pattern using class decorator
def singleton(cls):
    """Ensure only one instance of class exists"""
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Database:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        print(f"Database initialized with {connection_string}")

db1 = Database("localhost:5432")  # Database initialized
db2 = Database("localhost:3306")  # No output - returns existing instance
print(db1 is db2)  # True
```

```py
# Property-like class decorator
def immutable(cls):
    """Make class instances immutable after initialization"""
    original_setattr = cls.__setattr__
    original_delattr = cls.__delattr__
    
    def __setattr__(self, name, value):
        if hasattr(self, '_initialized'):
            raise AttributeError(f"Cannot modify immutable instance")
        original_setattr(self, name, value)
    
    def __delattr__(self, name):
        raise AttributeError(f"Cannot delete from immutable instance")
    
    cls.__setattr__ = __setattr__
    cls.__delattr__ = __delattr__
    
    # Mark initialization complete
    original_init = cls.__init__
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_setattr(self, '_initialized', True)
    
    cls.__init__ = __init__
    return cls

@immutable
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

point = Point(3, 4)
print(point.x)  # 3

try:
    point.x = 5  # AttributeError
except AttributeError as e:
    print(e)
```

#### Stacking Decorators

```py
@timer
@log_calls
@validate_positive
def multiply(x, y):
    return x * y

# Equivalent to:
# multiply = timer(log_calls(validate_positive(multiply)))

# Decorators are applied from bottom to top
print(multiply(5, 3))
```

- Best practices for decorators
    - Always use `@functools.wraps` it preserves `__name__`, `__doc__`, etc.
    - Use `*args` and `**kwargs` for flexibility
    - Make decorators optional/configurable

### Generators

Generators are functions that return an iterator which yields values one at a time, enabling lazy evaluation and memory efficiency.

```py
# Basic generator using yield
def count_up_to(n):
    """Generator that counts from 1 to n"""
    count = 1
    while count <= n:
        yield count  # Yields value and pauses
        count += 1

# Using the generator
counter = count_up_to(5)
print(type(counter))  # <class 'generator'>

print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # 3

# Or iterate with for loop
for num in count_up_to(3):
    print(num)
# 1
# 2
# 3

# Generator vs Regular Function
def regular_range(n):
    """Returns list (all values in memory)"""
    result = []
    for i in range(n):
        result.append(i)
    return result

def generator_range(n):
    """Returns generator (values created on demand)"""
    for i in range(n):
        yield i

# Memory comparison
import sys
regular = regular_range(1000)
generator = generator_range(1000)

print(sys.getsizeof(regular))     # ~9000 bytes
print(sys.getsizeof(generator))   # ~128 bytes (much smaller!)
```

#### Generator Execution Flow

```py
def demo_generator():
    """Demonstrates generator execution flow"""
    print("Start")
    yield 1
    print("Between 1 and 2")
    yield 2
    print("Between 2 and 3")
    yield 3
    print("End")

gen = demo_generator()
print("Generator created")
# Generator created

print(next(gen))
# Start
# 1

print(next(gen))
# Between 1 and 2
# 2

print(next(gen))
# Between 2 and 3
# 3

try:
    print(next(gen))
    # End
    # StopIteration exception raised
except StopIteration:
    print("Generator exhausted")
```

#### Practical Generator Examples

```py
# 1. Fibonacci sequence
def fibonacci_generator():
    """Infinite Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get first 10 Fibonacci numbers
fib = fibonacci_generator()
first_10 = [next(fib) for _ in range(10)]
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# 2. File reading (memory efficient)
def read_large_file(file_path):
    """Read file line by line without loading entire file"""
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Usage (processes one line at a time)
# for line in read_large_file('large_file.txt'):
#     process(line)

# 3. Data batching
def batch_data(data, batch_size):
    """Yield data in batches"""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

data = list(range(10))
for batch in batch_data(data, 3):
    print(batch)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]

# 4. Infinite sequence with conditions
def numbers_divisible_by(n):
    """Infinite sequence of numbers divisible by n"""
    num = n
    while True:
        yield num
        num += n

# Get first 5 numbers divisible by 7
divisible_by_7 = numbers_divisible_by(7)
result = [next(divisible_by_7) for _ in range(5)]
print(result)  # [7, 14, 21, 28, 35]

# 5. Data transformation pipeline
def read_data():
    """Simulate data source"""
    for i in range(10):
        yield i

def filter_even(numbers):
    """Filter even numbers"""
    for n in numbers:
        if n % 2 == 0:
            yield n

def square(numbers):
    """Square numbers"""
    for n in numbers:
        yield n ** 2

# Pipeline: read → filter → square
pipeline = square(filter_even(read_data()))
print(list(pipeline))  # [0, 4, 16, 36, 64]
```

#### Generator Expressions

Generator expressions are like list comprehensions but with parentheses, creating generators instead of lists.

```py
# List comprehension (creates list in memory)
squares_list = [x ** 2 for x in range(10)]
print(type(squares_list))  # <class 'list'>
print(sys.getsizeof(squares_list))  # ~200 bytes

# Generator expression (creates generator)
squares_gen = (x ** 2 for x in range(10))
print(type(squares_gen))  # <class 'generator'>
print(sys.getsizeof(squares_gen))  # ~128 bytes

# Iterate over generator
for square in squares_gen:
    print(square, end=' ')
# 0 1 4 9 16 25 36 49 64 81

# Generator expressions in functions
sum_of_squares = sum(x ** 2 for x in range(10))  # No extra parentheses needed
print(sum_of_squares)  # 285

# Chaining generator expressions
numbers = range(20)
even = (x for x in numbers if x % 2 == 0)
squared = (x ** 2 for x in even)
result = list(squared)
print(result)  # [0, 4, 16, 36, 64, 100, 144, 196, 256, 324]
```

- When to use generator expressions:
    - Processing large datasets
    - one-time iteration
    - Memory efficiency matters

- When **not** to use generator expression and consider using list comprehension:
    - Need to iterate multiple times
    - Need indexing or slicing
    - small dataset

#### Advanced Generator Features

```py
# 1. Generator.send() - Send values into generator
def echo_generator():
    """Generator that echoes sent values"""
    while True:
        received = yield
        print(f"Received: {received}")

gen = echo_generator()
next(gen)  # Prime the generator
gen.send("Hello")  # Received: Hello
gen.send("World")  # Received: World
```

```py
# 2. Generator.throw() - Throw exceptions into generator
def error_handler():
    """Generator that handles exceptions"""
    while True:
        try:
            value = yield
            print(f"Processing: {value}")
        except ValueError as e:
            print(f"Caught ValueError: {e}")

gen = error_handler()
next(gen)  # Prime
gen.send(42)  # Processing: 42
gen.throw(ValueError, "Invalid input")  # Caught ValueError: Invalid input
```

```py
# 3. Generator.close() - Close generator
def my_generator():
    try:
        while True:
            yield "value"
    finally:
        print("Generator closed - cleanup performed")

gen = my_generator()
print(next(gen))  # value
gen.close()  # Generator closed - cleanup performed

try:
    next(gen)  # StopIteration
except StopIteration:
    print("Generator is closed")
```

```py
# 4. yield from - Delegate to sub-generator
def sub_generator():
    yield 1
    yield 2

def main_generator():
    yield from sub_generator()  # Delegate
    yield 3

print(list(main_generator()))  # [1, 2, 3]

# Practical example: Flattening nested structures
def flatten(nested_list):
    """Recursively flatten nested list"""
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten(item)  # Recursive delegation
        else:
            yield item

nested = [1, [2, 3, [4, 5]], 6, [7, [8, 9]]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### Generator Use Cases and Performance

```py
import time

# Use Case 1: Processing large log files
def process_logs(log_file):
    """Memory-efficient log processing"""
    with open(log_file) as f:
        for line in f:
            if 'ERROR' in line:
                yield line.strip()

# Only loads matching lines into memory
# for error in process_logs('app.log'):
#     handle_error(error)
```

```py
# Use Case 2: Streaming data
def data_stream():
    """Simulate real-time data stream"""
    while True:
        # In real scenario: fetch from API, sensor, etc.
        data = fetch_next_data()
        yield data

# Process data as it arrives
# for data_point in data_stream():
#     process(data_point)
```

```py
# Performance comparison
def list_approach(n):
    """Load all data into list"""
    return [i ** 2 for i in range(n)]

def generator_approach(n):
    """Generate data on demand"""
    return (i ** 2 for i in range(n))

# Memory test
n = 1_000_000

# List: ~8MB for 1 million integers
start = time.time()
list_data = list_approach(n)
first_10 = list_data[:10]
print(f"List time: {time.time() - start:.4f}s")

# Generator: Constant memory
start = time.time()
gen_data = generator_approach(n)
first_10 = [next(gen_data) for _ in range(10)]
print(f"Generator time: {time.time() - start:.4f}s")
```

### Iterators and the Iterator Protocol

An iterator is an object that implements the iterator protocol:

- `__iter__()`: Returns the iterator object itself
- `__next__()`: Returns the next item or raises `StopIteration`

```py
# Built-in iterators
my_list = [1, 2, 3]
iterator = iter(my_list)  # Get iterator from iterable

print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3

try:
    print(next(iterator))  # StopIteration
except StopIteration:
    print("No more items")

# for loops use iterators under the hood
for item in [1, 2, 3]:
    print(item)

# Equivalent to:
iterator = iter([1, 2, 3])
while True:
    try:
        item = next(iterator)
        print(item)
    except StopIteration:
        break
```

#### Creating Custom Iterators

```py
class CountDown:
    """Iterator that counts down from start to 0"""
    
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        """Return iterator object (self)"""
        return self
    
    def __next__(self):
        """Return next value or raise StopIteration"""
        if self.current <= 0:
            raise StopIteration
        
        self.current -= 1
        return self.current + 1

# Using custom iterator
countdown = CountDown(5)
for num in countdown:
    print(num)
# 5, 4, 3, 2, 1

# Can only iterate once (stateful)
print(list(countdown))  # [] (already exhausted)

# Need to create new iterator
countdown2 = CountDown(3)
print(list(countdown2))  # [3, 2, 1]
```

- **Iterable vs Iterator**
    - **Iterable**: Object that can return an iterator (`__iter__`)
    - **Iterator**: Object with `__iter__` and `__next__`

| Aspect | Iterable | Iterator |
| :--- | :--- | :--- |
| **Definition** | Can be iterated over | Does the actual iteration |
| **Methods** | `__iter__()` | `__iter__()` and `__next__()` |
| **Multiple iterations** | Yes (creates new iterator) | No (stateful, one-time use) |
| **Examples** | list, dict, set, string, file | `iter(list)`, generator, `range` iterator |
| **`__iter__` returns** | New iterator object | `self` |

#### Practical Iterator example

```py
# 1. Fibonacci Iterator
class FibonacciIterator:
    """Iterator for Fibonacci sequence up to n terms"""
    
    def __init__(self, n):
        self.n = n
        self.count = 0
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count >= self.n:
            raise StopIteration
        
        value = self.a
        self.a, self.b = self.b, self.a + self.b
        self.count += 1
        return value

fib = FibonacciIterator(10)
print(list(fib))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```py
# 2. Reverse Iterator
class ReverseIterator:
    """Iterate over sequence in reverse"""
    
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        
        self.index -= 1
        return self.data[self.index]

reverse = ReverseIterator([1, 2, 3, 4, 5])
print(list(reverse))  # [5, 4, 3, 2, 1]
```

```py
# 3. Cyclic Iterator
class CyclicIterator:
    """Cycle through sequence indefinitely"""
    
    def __init__(self, data):
        self.data = data
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.data:
            raise StopIteration
        
        value = self.data[self.index]
        self.index = (self.index + 1) % len(self.data)
        return value

cycle = CyclicIterator(['A', 'B', 'C'])
print([next(cycle) for _ in range(10)])
# ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']
```

```py
# 4. File Line Iterator with Context
class FileLineIterator:
    """Iterator over file lines with automatic cleanup"""
    
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __iter__(self):
        self.file = open(self.filename, 'r')
        return self
    
    def __next__(self):
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()

# Usage (automatically closes file when iteration completes)
# for line in FileLineIterator('data.txt'):
#     process(line)
```

```py
# 5. Custom Range with Step
class CustomRange:
    """Range with custom step and predicates"""
    
    def __init__(self, start, end, step=1, predicate=None):
        self.start = start
        self.end = end
        self.step = step
        self.predicate = predicate or (lambda x: True)
    
    def __iter__(self):
        current = self.start
        while current < self.end:
            if self.predicate(current):
                yield current
            current += self.step

# Only even numbers between 0 and 20
even_range = CustomRange(0, 20, step=1, predicate=lambda x: x % 2 == 0)
print(list(even_range))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
```

#### Iterator Tools from `itertools`

```py
import itertools

# 1. itertools.count() - Infinite counter
counter = itertools.count(start=10, step=2)
print([next(counter) for _ in range(5)])  # [10, 12, 14, 16, 18]

# 2. itertools.cycle() - Cycle through sequence
colors = itertools.cycle(['red', 'green', 'blue'])
print([next(colors) for _ in range(7)])
# ['red', 'green', 'blue', 'red', 'green', 'blue', 'red']

# 3. itertools.repeat() - Repeat value
repeat_five = itertools.repeat(5, times=3)
print(list(repeat_five))  # [5, 5, 5]

# 4. itertools.chain() - Chain multiple iterables
combined = itertools.chain([1, 2], [3, 4], [5, 6])
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# 5. itertools.islice() - Slice iterator
numbers = itertools.count()  # Infinite
first_10_evens = itertools.islice(
    (x for x in numbers if x % 2 == 0), 
    10
)
print(list(first_10_evens))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

# 6. itertools.takewhile() - Take while condition is true
numbers = [1, 2, 3, 4, 5, 1, 2, 3]
result = itertools.takewhile(lambda x: x < 4, numbers)
print(list(result))  # [1, 2, 3]

# 7. itertools.dropwhile() - Drop while condition is true
numbers = [1, 2, 3, 4, 5, 1, 2, 3]
result = itertools.dropwhile(lambda x: x < 4, numbers)
print(list(result))  # [4, 5, 1, 2, 3]

# 8. itertools.groupby() - Group consecutive elements
data = [1, 1, 2, 2, 2, 3, 3, 1, 1]
groups = itertools.groupby(data)
result = [(key, list(group)) for key, group in groups]
print(result)  # [(1, [1, 1]), (2, [2, 2, 2]), (3, [3, 3]), (1, [1, 1])]

# 9. itertools.combinations() - All combinations
items = ['A', 'B', 'C']
combos = itertools.combinations(items, 2)
print(list(combos))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 10. itertools.permutations() - All permutations
items = ['A', 'B', 'C']
perms = itertools.permutations(items, 2)
print(list(perms))  # [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# 11. itertools.product() - Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
products = itertools.product(colors, sizes)
print(list(products))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# 12. Real-world example: Pagination
def paginate(data, page_size):
    """Paginate data using islice"""
    iterator = iter(data)
    while True:
        page = list(itertools.islice(iterator, page_size))
        if not page:
            break
        yield page

data = range(25)
for page_num, page in enumerate(paginate(data, 10), start=1):
    print(f"Page {page_num}: {page}")
# Page 1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# Page 2: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
# Page 3: [20, 21, 22, 23, 24]
```

### Advanced Functional Patterns

#### Currying and Partial Application

```py
from functools import partial

# Manual currying
def add(x):
    """Curried addition"""
    def add_y(y):
        def add_z(z):
            return x + y + z
        return add_z
    return add_y

result = add(1)(2)(3)
print(result)  # 6

# Automatic currying decorator
def curry(func):
    """Convert function to curried form"""
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more: curried(*(args + more))
    return curried

@curry
def multiply(x, y, z):
    return x * y * z

print(multiply(2)(3)(4))  # 24
print(multiply(2, 3)(4))  # 24
print(multiply(2)(3, 4))  # 24
print(multiply(2, 3, 4))  # 24
```

```py
# Partial application (more common in Python)
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125
```

```py
# Practical example: Logger with context
def log(level, message, context=None):
    """Log message with level and context"""
    ctx = f" [{context}]" if context else ""
    print(f"[{level}]{ctx} {message}")

# Create specialized loggers
error_log = partial(log, "ERROR")
info_log = partial(log, "INFO")
debug_log = partial(log, "DEBUG", context="MyApp")

error_log("Connection failed")  # [ERROR] Connection failed
info_log("Process started")     # [INFO] Process started
debug_log("Variable x = 5")     # [DEBUG] [MyApp] Variable x = 5
```

#### Lazy Evaluation Patterns

```py
# 1. Lazy property evaluation
class ExpensiveResource:
    """Compute expensive value only when accessed"""
    
    def __init__(self):
        self._value = None
    
    @property
    def value(self):
        """Lazy evaluation"""
        if self._value is None:
            print("Computing expensive value...")
            self._value = sum(range(1000000))  # Expensive computation
        return self._value

resource = ExpensiveResource()
print("Resource created")  # No computation yet
print(resource.value)      # Computing expensive value... 499999500000
print(resource.value)      # 499999500000 (cached, no recomputation)
```

```py
# 2. Lazy sequences with generators
def lazy_fibonacci():
    """Infinite lazy Fibonacci sequence"""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Only computed when needed
fib = lazy_fibonacci()
first_10 = [next(fib) for _ in range(10)]
print(first_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
```

```py
# 3. Lazy evaluation with itertools
import itertools

# Infinite sequence of squares, but only compute first 5
squares = (x ** 2 for x in itertools.count())
first_5_squares = list(itertools.islice(squares, 5))
print(first_5_squares)  # [0, 1, 4, 9, 16]
```

```py
# 4. Thunks (delayed computation)
class Thunk:
    """Wrapper for delayed computation"""
    
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._computed = False
        self._value = None
    
    def force(self):
        """Force evaluation"""
        if not self._computed:
            self._value = self.func(*self.args, **self.kwargs)
            self._computed = True
        return self._value

def expensive_computation(x, y):
    print("Computing...")
    return x ** y

# Create thunk (no computation yet)
thunk = Thunk(expensive_computation, 2, 20)
print("Thunk created")  # No output

# Force evaluation when needed
result = thunk.force()  # Computing...
print(result)           # 1048576

# Subsequent calls return cached value
result2 = thunk.force()  # No "Computing..." output
print(result2)           # 1048576
```

#### Monadic Patterns

monads are a design pattern used to structure computations by chaining together functions that return the same type of container or context, such as a `List` or `Maybe`.

They provide a uniform interface to handle values within a context, abstracting away concerns like side effects, potential errors, or asynchronous operations to make code more declarative and robust. Key operations include a "**unit**" function to lift a value into the monad and a "**bind**" operation to sequence computations, which often flattens nested contexts.

```py
class Maybe:
    """Optional/Maybe monad for handling None values"""
    
    def __init__(self, value):
        self.value = value
    
    def bind(self, func):
        """Chain operations, short-circuit on None"""
        if self.value is None:
            return Maybe(None)
        return Maybe(func(self.value))
    
    def map(self, func):
        """Transform value if present"""
        return self.bind(func)
    
    def get_or_else(self, default):
        """Get value or default"""
        return self.value if self.value is not None else default
    
    def __repr__(self):
        return f"Maybe({self.value})"

# Example usage
def divide(x, y):
    """Safe division"""
    return x / y if y != 0 else None

def add_ten(x):
    return x + 10

def double(x):
    return x * 2

# Chain operations safely
result = Maybe(20).bind(lambda x: divide(x, 2)).map(add_ten).map(double)
print(result)  # Maybe(40.0)

# Short-circuit on None
result = Maybe(20).bind(lambda x: divide(x, 0)).map(add_ten).map(double)
print(result)  # Maybe(None)

# Get final value with default
print(result.get_or_else(0))  # 0
```

```py
# Real-world example: Safe data access
class SafeDict(dict):
    """Dictionary with safe chaining"""
    
    def get_maybe(self, key):
        return Maybe(self.get(key))

data = SafeDict({
    'user': {
        'profile': {
            'email': 'user@example.com'
        }
    }
})

# Safe nested access
email = (
    data.get_maybe('user')
    .bind(lambda u: u.get('profile'))
    .bind(lambda p: p.get('email'))
    .get_or_else('no-email@example.com')
)
print(email)  # user@example.com

# Handles missing keys gracefully
city = (
    data.get_maybe('user')
    .bind(lambda u: u.get('address'))
    .bind(lambda a: a.get('city'))
    .get_or_else('Unknown')
)
print(city)  # Unknown
```

#### Functional Data Structures

```py
from typing import List, Tuple, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

# Immutable linked list
@dataclass(frozen=True)
class Cons(Generic[T]):
    """Cons cell for functional list"""
    head: T
    tail: 'List[T]'

class Nil:
    """Empty list"""
    pass

def cons_list(*items):
    """Create functional list from items"""
    result = Nil()
    for item in reversed(items):
        result = Cons(item, result)
    return result

def cons_map(func, lst):
    """Map function over functional list"""
    if isinstance(lst, Nil):
        return Nil()
    return Cons(func(lst.head), cons_map(func, lst.tail))

def cons_filter(pred, lst):
    """Filter functional list"""
    if isinstance(lst, Nil):
        return Nil()
    if pred(lst.head):
        return Cons(lst.head, cons_filter(pred, lst.tail))
    return cons_filter(pred, lst.tail)

# Usage
my_list = cons_list(1, 2, 3, 4, 5)
doubled = cons_map(lambda x: x * 2, my_list)
evens = cons_filter(lambda x: x % 2 == 0, my_list)

print(my_list)   # Cons(head=1, tail=Cons(head=2, tail=...))
print(doubled)   # Cons(head=2, tail=Cons(head=4, tail=...))
```

### Optimization Techniques For Functional Programming in Python

1. Use built-in functions (C-optimized)

```py
# ❌ Slow
def sum_manual(numbers):
    total = 0
    for n in numbers:
        total += n
    return total

# ✅ Fast
def sum_builtin(numbers):
    return sum(numbers)
```

2. Use list comprehension over map/filter when converting to list

```py
numbers = range(1000)

# Slower (two passes)
result = list(map(lambda x: x ** 2, filter(lambda x: x % 2 == 0, numbers)))

# Faster (one pass)
result = [x ** 2 for x in numbers if x % 2 == 0]
```

3. Use generator for chained operations

```py
# ❌ Multiple intermediate lists
def process_data(data):
    filtered = [x for x in data if x > 0]
    mapped = [x ** 2 for x in filtered]
    return [x for x in mapped if x < 100]

# ✅ Single pass with generator
def process_data_efficient(data):
    return [x ** 2 for x in data if x > 0 and x ** 2 < 100]
```

4. Use itertools for combinations

```py
import itertools

# ❌ Slow for large n
def combinations_manual(items, r):
    result = []
    # ... complex nested loops
    return result

# ✅ Fast (C implementation)
def combinations_fast(items, r):
    return list(itertools.combinations(items, r))
```

5. Cache expensive computations

```py
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Fast due to caching
```

### Example Codes

#### 1. Data Pipeline Processing

```python
from typing import List, Dict, Any
import json

def create_pipeline(*functions):
    """Create a data processing pipeline"""
    def pipeline(data):
        return reduce(lambda result, func: func(result), functions, data)
    return pipeline

# Data cleaning functions
def parse_json(data: str) -> List[Dict]:
    return json.loads(data)

def filter_valid_records(records: List[Dict]) -> List[Dict]:
    return [r for r in records if r.get('id') and r.get('amount')]

def normalize_amounts(records: List[Dict]) -> List[Dict]:
    return [
        {**r, 'amount': float(r['amount']) if isinstance(r['amount'], str) else r['amount']}
        for r in records
    ]

def add_categories(records: List[Dict]) -> List[Dict]:
    def categorize(amount):
        if amount < 100:
            return 'small'
        elif amount < 1000:
            return 'medium'
        else:
            return 'large'
    
    return [
        {**r, 'category': categorize(r['amount'])}
        for r in records
    ]

def calculate_statistics(records: List[Dict]) -> Dict[str, Any]:
    amounts = [r['amount'] for r in records]
    return {
        'records': records,
        'stats': {
            'count': len(amounts),
            'total': sum(amounts),
            'average': sum(amounts) / len(amounts) if amounts else 0,
            'max': max(amounts) if amounts else 0,
            'min': min(amounts) if amounts else 0
        }
    }

# Create and use pipeline
process_financial_data = create_pipeline(
    parse_json,
    filter_valid_records,
    normalize_amounts,
    add_categories,
    calculate_statistics
)

# Example usage
raw_data = '''[
    {"id": 1, "amount": "150.50"},
    {"id": 2, "amount": 2500},
    {"amount": 100},
    {"id": 4, "amount": "75.25"}
]'''

result = process_financial_data(raw_data)
print(f"Processed {result['stats']['count']} records")
print(f"Total amount: ${result['stats']['total']:.2f}")
```

#### 2. Event Stream Processing

```python
from typing import Callable, List
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    type: str
    data: Dict[str, Any]
    timestamp: datetime

class EventStream:
    """Functional event stream processor"""
    def __init__(self, events: List[Event] = None):
        self.events = events or []
    
    def filter(self, predicate: Callable[[Event], bool]) -> 'EventStream':
        return EventStream([e for e in self.events if predicate(e)])
    
    def map(self, transformer: Callable[[Event], Event]) -> 'EventStream':
        return EventStream([transformer(e) for e in self.events])
    
    def reduce(self, reducer: Callable[[Any, Event], Any], initial: Any) -> Any:
        return reduce(reducer, self.events, initial)
    
    def group_by(self, key_func: Callable[[Event], str]) -> Dict[str, List[Event]]:
        result = {}
        for event in self.events:
            key = key_func(event)
            if key not in result:
                result[key] = []
            result[key].append(event)
        return result

# Usage example
events = [
    Event("login", {"user_id": 1}, datetime(2024, 1, 1, 9, 0)),
    Event("purchase", {"user_id": 1, "amount": 100}, datetime(2024, 1, 1, 9, 30)),
    Event("login", {"user_id": 2}, datetime(2024, 1, 1, 10, 0)),
    Event("purchase", {"user_id": 2, "amount": 50}, datetime(2024, 1, 1, 10, 15)),
    Event("logout", {"user_id": 1}, datetime(2024, 1, 1, 11, 0)),
]

stream = EventStream(events)

# Get all purchase events
purchases = stream.filter(lambda e: e.type == "purchase")

# Calculate total revenue
total_revenue = purchases.reduce(
    lambda acc, e: acc + e.data.get("amount", 0),
    0
)

print(f"Total revenue: ${total_revenue}")

# Group events by type
events_by_type = stream.group_by(lambda e: e.type)
for event_type, type_events in events_by_type.items():
    print(f"{event_type}: {len(type_events)} events")
```

#### 3. Configuration Management

```python
from functools import reduce
from typing import Dict, Any

def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Functionally merge multiple configuration dictionaries"""
    def deep_merge(dict1, dict2):
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return reduce(deep_merge, configs, {})

def create_config_builder():
    """Functional config builder pattern"""
    def builder(config=None):
        config = config or {}
        
        def with_database(host, port, name):
            return builder(merge_configs(config, {
                'database': {'host': host, 'port': port, 'name': name}
            }))
        
        def with_cache(enabled, ttl=300):
            return builder(merge_configs(config, {
                'cache': {'enabled': enabled, 'ttl': ttl}
            }))
        
        def with_logging(level, format_string=None):
            return builder(merge_configs(config, {
                'logging': {'level': level, 'format': format_string}
            }))
        
        def build():
            return config
        
        builder.with_database = with_database
        builder.with_cache = with_cache
        builder.with_logging = with_logging
        builder.build = build
        
        return builder
    
    return builder()

# Usage
config = (create_config_builder()
    .with_database('localhost', 5432, 'myapp')
    .with_cache(True, ttl=600)
    .with_logging('INFO', '%(asctime)s - %(message)s')
    .build())

print(config)
```

### Common Patterns

```python
# 1. Pipeline pattern
pipeline = compose(
    validate,
    transform,
    enrich,
    save
)

# 2. Maybe/Option pattern for null safety
def safe_operation(value):
    return Maybe(value).map(process).bind(validate).get_or_default(default_value)

# 3. Reducer pattern for aggregation
result = reduce(combine, map(transform, filter(predicate, data)), initial)

# 4. Builder pattern (functional style)
config = (ConfigBuilder()
    .with_option_a(value1)
    .with_option_b(value2)
    .build())
```

### Testing Functional Code

```python
import unittest

class TestFunctionalCode(unittest.TestCase):
    def test_pure_function(self):
        """Pure functions are easy to test"""
        self.assertEqual(calculate_tax(100, 0.1), 10)
        self.assertEqual(calculate_tax(100, 0.1), 10)  # Always same result
    
    def test_composition(self):
        """Test composed functions"""
        f = compose(lambda x: x * 2, lambda x: x + 1)
        self.assertEqual(f(5), 12)  # (5 + 1) * 2
    
    def test_immutability(self):
        """Ensure data isn't mutated"""
        original = [1, 2, 3]
        result = add_item_immutable(original, 4)
        self.assertEqual(original, [1, 2, 3])  # Unchanged
        self.assertEqual(result, [1, 2, 3, 4])
```

- Start with **pure functions** and **immutability**
- Use **higher-order functions** for **abstraction**
- Compose simple functions into complex behaviors

---

[Remaining python sections](./Python%20III.md)
