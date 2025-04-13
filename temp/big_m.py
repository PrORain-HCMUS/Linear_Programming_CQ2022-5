M = 1000  # Hệ số M lớn dùng trong phương pháp Big-M

# def parse_input():
#     # Hàm mục tiêu: Max z = 2x1 - x2 - 2x3
#     c = [2, -1, -2]

#     # Ma trận hệ số A
#     A = [
#         [1, 1, -1],
#         [1, 2, 1]
#     ]

#     # Vế phải
#     b = [-1, 2]

#     # Dấu ràng buộc
#     signs = ['=', '=']

#     return c, A, b, signs

def parse_input():
    # Max Z = 7x1 + x2 - 4x3 
    # Ràng buộc:
    # -6x1 + 4x2 + 5x3 >= -20
    # x1 + 2x2 - x3 = 8
    # 3x1 + 2x2 - x3 <= -8
    c = [7, 1, -4]
    A = [
        [-6, 4, 5],
        [1, 2, 1],
        [3, 2, -1]
    ]
    b = [-20, 8, -8]
    signs = ['>=', '=', '<=']
    return c, A, b, signs


def preprocess_constraints(A, b, signs):
    new_A = []
    new_b = []
    new_signs = []

    for i in range(len(A)):
        row = A[i]
        rhs = b[i]
        sign = signs[i]

        # Đưa RHS về không âm
        if rhs < 0:
            row = [-x for x in row]
            rhs = -rhs
            if sign == '<=':
                sign = '>='
            elif sign == '>=':
                sign = '<='

        new_A.append(row)
        new_b.append(rhs)
        new_signs.append(sign)

    return new_A, new_b, new_signs


def build_bigM_tableau(c, A, b, signs):
    A, b, signs = preprocess_constraints(A, b, signs)

    num_orig_vars = len(c)
    tableau = []
    basis = []
    basis_cost = []
    var_names = [f"x{i+1}" for i in range(num_orig_vars)]
    c_extended = c[:]  # Bắt đầu với hệ số gốc

    for i, (row, sign) in enumerate(zip(A, signs)):
        row_extended = row[:]

        if sign == '<=':
            # Thêm biến slack
            row_extended += [0] * (len(var_names) - len(row_extended))
            row_extended.append(1)
            var_names.append(f"x{len(var_names)+1}")
            c_extended.append(0)
            basis.append(len(var_names) - 1)
            basis_cost.append(0)

        elif sign == '>=':
            # Thêm biến surplus và artificial
            row_extended += [0] * (len(var_names) - len(row_extended))
            row_extended.append(-1)
            var_names.append(f"x{len(var_names)+1}")
            c_extended.append(0)

            row_extended.append(1)
            var_names.append(f"x{len(var_names)+1}")
            c_extended.append(-M)
            basis.append(len(var_names) - 1)
            basis_cost.append(-M)

        elif sign == '=':
            # Thêm artificial
            row_extended += [0] * (len(var_names) - len(row_extended))
            row_extended.append(1)
            var_names.append(f"x{len(var_names)+1}")
            c_extended.append(-M)
            basis.append(len(var_names) - 1)
            basis_cost.append(-M)

        tableau.append(row_extended)

    # Đệm 0 để làm đều các dòng
    total_vars = len(var_names)
    for row in tableau:
        row += [0] * (total_vars - len(row))

    # Cách tính z_row HOÀN TOÀN MỚI dựa trên định nghĩa
    z_row = [0] * total_vars
    rhs_z = 0
    
    # Tính RHS của hàng Z: Σ(cBi·bi)
    for i in range(len(tableau)):
        rhs_z += basis_cost[i] * b[i]
    
    # Với mỗi biến j, tính Δj = Σ(cBi·aij) - cj
    for j in range(total_vars):
        # Tính Σ(cBi·aij)
        sum_cb_aij = 0
        for i in range(len(tableau)):
            sum_cb_aij += basis_cost[i] * tableau[i][j]
        
        # Trừ đi cj
        z_row[j] = sum_cb_aij - c_extended[j]

    return basis, basis_cost, b, tableau, z_row, rhs_z, var_names, c_extended


def print_bigM_tableau(basis, basis_cost, b, tableau, z_row, rhs_z, var_names, c_extended):
    print("\nBảng đơn hình ban đầu (phương pháp Big-M):\n")

    num_vars = len(var_names)
    header = ["Biến cơ sở", "Hệ số", "Hệ số RHS"] + var_names
    print(" | ".join(f"{h:^12}" for h in header))
    print("-" * (14 * len(header)))

    # Hiển thị hệ số hàm mục tiêu (bao gồm cả artificial variables)
    c_row = [" " * 12, " " * 12, " " * 12] + [f"{coef:^12}" for coef in c_extended]
    print(" | ".join(c_row))
    print("-" * (14 * len(header)))

    for i in range(len(basis)):
        var = var_names[basis[i]]
        coef = basis_cost[i]
        row_vals = tableau[i]
        row_str = [f"{var:^12}", f"{coef:^12}", f"{b[i]:^12}"] + [f"{v:^12}" for v in row_vals]
        print(" | ".join(row_str))

    # Hiển thị toàn bộ z_row
    z_row_str = ["Z".center(12), " ".center(12), f"{rhs_z:^12}"] + [f"{v:^12}" for v in z_row]
    print("-" * (14 * len(header)))
    print(" | ".join(z_row_str))


# ========== CHẠY THỬ ==========
c, A, b, signs = parse_input()
basis, basis_cost, b, tableau, z_row, rhs_z, var_names, c_extended = build_bigM_tableau(c, A, b, signs)
print_bigM_tableau(basis, basis_cost, b, tableau, z_row, rhs_z, var_names, c_extended)