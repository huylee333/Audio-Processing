# Danh sách việc cần làm (Lab 2)

Dựa theo mục 4 (Nội dung thực hành bắt buộc) trong hướng dẫn Lab 2, dưới đây là các công việc bạn cần thực hiện tuần tự:

## A. Thu dữ liệu và kiểm tra chất lượng
- [ ] Tạo vocabulary 5–10 từ tách rời; mỗi từ tối thiểu 5 lần lặp.
- [ ] Đưa tất cả file về định dạng WAV mono 16 kHz.
- [ ] Vẽ waveform của ít nhất 3 từ; kiểm tra clipping và độ dài silence đầu/cuối.
- [ ] Chia train/test trước khi chọn template, tránh dùng cùng file cho cả train và test.

## B. Đặc trưng miền thời gian
- [ ] Chia frame 25 ms, hop 10 ms.
- [ ] Tính short-time energy/RMS và ZCR cho ít nhất 3 file.
- [ ] Vẽ energy và ZCR theo thời gian trên cùng trục thời gian với waveform.
- [ ] Mô tả sự khác biệt giữa silence, voiced và unvoiced.

## C. Endpoint detection
- [ ] Xây dựng hàm tìm start/end frame dựa trên log-energy (có thể kết hợp ZCR để tinh chỉnh).
- [ ] Xuất file speech đã trim và so sánh thời lượng trước/sau.
- [ ] Kiểm tra ít nhất một từ bắt đầu/kết thúc bằng phụ âm năng lượng thấp; điều chỉnh margin để không cắt mất âm.

## D. MFCC
- [ ] Cài đặt tiền nhấn (Pre-emphasis) với α=0.97 (hoặc giải thích nếu không dùng).
- [ ] Dùng Hamming window; n_mels=24; n_mfcc=13; giữ các tham số giống nhau cho cả train và test.
- [ ] Vẽ ma trận MFCC của ít nhất 2 từ khác nhau.
- [ ] Trả lời câu hỏi: Giải thích tại sao mỗi file có số frame khác nhau nhưng số chiều MFCC/frame là cố định?

## E. DTW
- [ ] Cài đặt hàm tính Euclidean local distance.
- [ ] Tự lập trình thuật toán dynamic programming cho DTW (không gọi thư viện có sẵn cho phần bắt buộc).
- [ ] Vẽ local-distance matrix và optimal path của: (i) 2 lượt phát âm cùng một từ, (ii) 2 từ khác nhau.
- [ ] So sánh DTW_norm giữa hai trường hợp trên và giải thích.

## F. Bộ nhận dạng nearest-template
- [ ] Với mỗi test file: Trích feature → tính DTW đến toàn bộ templates → chọn nhãn có khoảng cách nhỏ nhất.
- [ ] In bảng top-3 nhãn gần nhất và khoảng cách tương ứng cho ít nhất 5 test file.
- [ ] (Tùy chọn) Reject: nếu khoảng cách tốt nhất > θ, trả về 'unknown' (phải mô tả cách chọn θ nếu dùng).

## G. Đánh giá và thí nghiệm
- [ ] Tính accuracy và vẽ confusion matrix cho bộ nhận dạng.
- [ ] Thực hiện thí nghiệm bắt buộc 1: So sánh giữa có endpoint detection và không có endpoint detection.
- [ ] Thực hiện thí nghiệm bắt buộc 2: So sánh MFCC 13 hệ số với MFCC + Δ (vận tốc).
- [ ] (Nếu làm nhóm/có dữ liệu 2 người): Thí nghiệm same-speaker vs. cross-speaker.
