class Calculator:
    def add(self, x, y):
        # Check if x and y are strings representing valid integers
        if isinstance(x, str) and x.isdigit():
            x = int(x)
        if isinstance(y, str) and y.isdigit():
            y = int(y)

        # Check for valid types
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("Invalid input type")

        return x + y
