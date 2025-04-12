import sympy as sp

# Khởi tạo biến
x, x_prime = sp.symbols('x x_prime')

# Tọa độ điểm P
y_P = x**2

# Hệ số góc của BP
m_BP = (x**2 - 100) / (x - 10)

# Phương trình BP: y = m(x' - 10) + 100
y_BP = m_BP * (x_prime - 10) + 100

# Phương trình tiếp tuyến tại A: y = 4x' - 4
y_tangent = 4 * x_prime - 4

# Giao điểm C: giải phương trình y_BP = y_tangent
eq_C = sp.Eq(y_BP, y_tangent)
x_C = sp.solve(eq_C, x_prime)[0]
y_C = y_tangent.subs(x_prime, x_C)

# Tọa độ điểm C đã tìm được
C = (x_C.simplify(), y_C.simplify())

# Tìm hình chiếu D của P xuống đường y = 4x - 4
# Đường vuông góc với d: vector pháp tuyến (4, -1)
# Phương trình đường vuông góc qua P: y - x^2 = -1/4 (x' - x)
x_d = sp.Symbol('x_d')
y_d = -1/4 * (x_d - x) + x**2

# Giao điểm với d: y = 4x_d - 4
eq_D = sp.Eq(y_d, 4 * x_d - 4)
x_D = sp.solve(eq_D, x_d)[0]
y_D = 4 * x_D - 4

# Tọa độ D
D = (x_D.simplify(), y_D.simplify())

# Trung điểm AD
A = (2, 4)
mid_AD = ((2 + x_D) / 2, (4 + y_D) / 2)

# Đặt điều kiện: C là trung điểm của AD
cond_x = sp.Eq(C[0], mid_AD[0])
cond_y = sp.Eq(C[1], mid_AD[1])

# Giải hệ để tìm x (tọa độ hoành độ của P)
solution = sp.solve([cond_x, cond_y], x)
solution

