# Yêu cầu Project — Electronics for IT (PSO / IRS)

Tài liệu này tổng hợp **toàn bộ yêu cầu** của project theo từng giai đoạn.
Bản gốc: [`Project_ElectronicIT.pdf`](Project_ElectronicIT.pdf) (đề bài ban đầu),
[`Update_Project_Elect_IT.pdf`](Update_Project_Elect_IT.pdf) (yêu cầu mở rộng, 07/07/2026),
[`Project_PhaseShift_Model.pdf`](Project_PhaseShift_Model.pdf) (bài báo tham khảo).

---

## Giai đoạn 1 — Yêu cầu ban đầu (Tuần 1–4)

Thuật toán được chọn: **Particle Swarm Optimization (PSO)**.
Bài toán áp dụng: tối ưu pha phản xạ của hệ thống **IRS** (Intelligent
Reflecting Surface) theo mô hình phase-shift thực tế của bài báo tham khảo
(Abeywickrama *et al.*).

| Tuần | Nội dung |
|------|----------|
| 1 | Mô tả thuật toán PSO gốc, các bước thực hiện, ứng dụng, ví dụ minh họa, các biến thể và thuật toán liên quan |
| 2 | Mô hình hệ thống IRS và mô hình phase-shift thực tế (Eq. 5) |
| 3 | Cài đặt AO (Alternating Optimization) theo bài báo — baseline |
| 4 | Cài đặt PSO cho bài toán tối ưu pha θ, kết quả 1000 realizations, phân tích độ phức tạp PSO vs AO |

Mô hình tương đương (Eq. 5 của bài báo):

$$v_n = \beta_n(\theta_n)\, e^{j\theta_n},\qquad
\beta(\theta) = (1-\beta_{\min})\left(\frac{\sin(\theta-\phi)+1}{2}\right)^k + \beta_{\min}$$

với hằng số fit: $\beta_{\min}=0.2$, $k=1.6$, $\phi=0.43\pi$.

Mã nguồn giai đoạn này: [`src/phase_model/`](../src/phase_model/).

---

## Giai đoạn 2 — Yêu cầu mở rộng (Tuần 5, cập nhật 07/07/2026)

### Bài toán

**Không** tối ưu pha tương đương $\theta_n$ nữa, mà tối ưu **trực tiếp các
tham số linh kiện** của từng phần tử IRS:

$$\mathbf{x} = [L_{1,1}, L_{2,1}, C_1, R_1,\; \ldots,\; L_{1,N}, L_{2,N}, C_N, R_N]^T \in \mathbb{R}^{4N}$$

### Chuỗi tính toán cho mỗi nghiệm ứng viên

1. **Linh kiện → Trở kháng** (Eq. 3):

$$Z_n = \frac{j\omega L_{1,n}\left(j\omega L_{2,n} + \frac{1}{j\omega C_n} + R_n\right)}
{j\omega L_{1,n} + j\omega L_{2,n} + \frac{1}{j\omega C_n} + R_n}$$

2. **Trở kháng → Hệ số phản xạ vật lý** (Eq. 4): $v_n = \dfrac{Z_n - Z_0}{Z_n + Z_0}$, với $Z_0 = 377\,\Omega$.

3. **Vector phản xạ → Tốc độ truyền** (Eq. 6):

$$R_{SE} = \log_2\!\left(1 + \frac{\left|(\mathbf{v}^H \boldsymbol{\Phi} + \mathbf{h}_d^H)\mathbf{w}\right|^2}{\sigma^2}\right)$$

4. Dùng $R_{SE}$ làm hàm thích nghi (fitness); bài toán: $\max R_{SE}$
   (hoặc $\min -R_{SE}$).

### Ràng buộc vật lý (Eq. 6–9) và căn cứ lựa chọn

| Linh kiện | Khoảng | Căn cứ |
|-----------|--------|--------|
| $C_n$ | 0.47 – 2.35 pF | dải điều chỉnh varactor dùng trong bài báo gốc |
| $L_{1,n}$ | 0.5 – 5 nH | quanh giá trị bài báo (2.5 nH), dải chip inductor thực tế ở 2.4 GHz |
| $L_{2,n}$ | 0.1 – 2 nH | quanh giá trị bài báo (0.7 nH), điện cảm ký sinh lớp trên |
| $R_n$ | 0.5 – 5 Ω | bài báo khảo sát R ∈ {1, 2.5} Ω; 0.5 Ω là sàn tổn hao thực tế |

(Cài đặt tại [`src/rlc_model/config.py`](../src/rlc_model/config.py).)

### Nội dung bắt buộc trong báo cáo

- [x] Thuật toán tối ưu được chọn (PSO) và lý do
- [x] Cách biểu diễn nghiệm ứng viên (vector 4N chiều, chuẩn hóa [0,1])
- [x] Các biến tối ưu $L_{1,n}, L_{2,n}, C_n, R_n$
- [x] Hàm mục tiêu dựa trên $R_{SE}$
- [x] Ràng buộc vật lý của linh kiện + căn cứ
- [x] Cách tính linh kiện → $Z_n$ (Eq. 3) — `circuit.element_impedance`
- [x] Cách tính $Z_n$ → $v_n$ (Eq. 4) — `circuit.reflection_coefficient`
- [x] Kết quả tối ưu + phân tích ảnh hưởng của tham số linh kiện (exp5)
- [x] So sánh với AO tối ưu pha theo mô hình tương đương Eq. (5) (exp2–exp4)
- [x] So sánh khi một số linh kiện cố định (exp5: chỉ C; C,R; C,L1)

### Ánh xạ yêu cầu → thí nghiệm

| Yêu cầu | Script | Hình |
|---------|--------|------|
| Kiểm chứng mô hình mạch tương đương Eq.(5) suy ra được từ Eq.(3)–(4) | `exp1_circuit_verification.py` | `rlc_fig1`, `rlc_fig2` |
| Hội tụ PSO mức linh kiện vs AO | `exp2_convergence.py` | `rlc_fig3` |
| Tốc độ truyền vs khoảng cách AP–User | `exp3_rate_vs_distance.py` | `rlc_fig4` |
| Tốc độ truyền vs số phần tử N | `exp4_rate_vs_N.py` | `rlc_fig5` |
| So sánh khi cố định một số linh kiện + phân bố giá trị tối ưu | `exp5_component_analysis.py` | `rlc_fig6`, `rlc_fig7` |
