# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** Chưa xác nhận từ notebook.

**Runtime Colab:** Chưa xác nhận CPU/GPU.

**Python / PyTorch / Ultralytics:** Python và PyTorch chưa xác nhận; Ultralytics `8.4.145` theo metadata trong các file JSON.

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`.

**Thay đổi so với notebook nguồn:** Chưa xác nhận do chưa có notebook để đối chiếu.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn bằng chứng: `classification_predictions.json` và `visuals/classification_top5.png`, sample `traffic`.

**Record hạng 1:**

```json
{
  "class_id": 468,
  "class_name": "cab",
  "rank": 1,
  "score": 0.510915,
  "taxonomy_name": "ImageNet-1K"
}
```

Record cho biết mô hình chọn `cab` (taxi) là lớp có điểm cao nhất để mô tả toàn ảnh trong danh sách lớp của checkpoint. Đây là prediction cấp ảnh, không chỉ ra chiếc xe nào là taxi và không đếm số taxi. Ảnh thực tế có nhiều phương tiện, trong đó có nhiều xe buýt; vì vậy một nhãn duy nhất chưa mô tả đầy đủ cảnh giao thông.

Danh sách lớp xuất phát từ taxonomy của bộ dữ liệu huấn luyện và được bên xây dựng checkpoint sử dụng để định nghĩa đầu ra mô hình. Checkpoint này dùng `ImageNet-1K`; người dùng không thể chỉ đổi tên nhãn để mô hình nhận biết một lớp mới.

Cần giữ cả ID, tên lớp và tên taxonomy: ID phục vụ xử lý bằng máy, tên lớp giúp con người đọc hiểu, còn taxonomy xác định ngữ nghĩa và phạm vi của bộ nhãn. Cùng một ID trong hai taxonomy có thể mang ý nghĩa khác nhau.

Với ảnh có nhiều chủ thể, guideline cần quy định dùng một nhãn hay nhiều nhãn; nếu dùng một nhãn thì phải có tiêu chí chọn chủ thể chính và cách xử lý khi không thể lựa chọn rõ ràng. Không tự động lấy top-1 của mô hình làm nhãn chuẩn.

Model score thể hiện mức điểm mô hình gán cho prediction, không phải chất lượng ground truth hoặc bằng chứng prediction chắc chắn đúng. Ground truth cần được con người gán và kiểm tra theo guideline. Score `0.510915` không chứng minh ảnh đã được gán nhãn đúng với tỷ lệ 51,09%.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn bằng chứng: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

**Một record:**

```json
{
  "class_name": "person",
  "score": 0.912625,
  "bbox_xyxy": [385.33, 69.24, 498.92, 348.92],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}
```

Ảnh gốc có kích thước `640 × 427` pixel. Box được ghi theo định dạng `[x_min, y_min, x_max, y_max]`, với gốc tọa độ tại góc trên bên trái. Box bao quanh người đứng ở nửa phải ảnh, từ gần đầu đến chân; góc trên trái ở `(385.33, 69.24)` và góc dưới phải ở `(498.92, 348.92)`. Chiều rộng và chiều cao được trích nguyên từ JSON; sai khác 0,01 pixel khi trừ các tọa độ đã làm tròn có thể xuất hiện do làm tròn riêng từng trường.

**So sánh số prediction ở ba threshold:**

Nguồn bằng chứng: output ô so sánh threshold trong notebook đã chạy trên Colab, sample `kitchen`. File JSON lưu kết quả tại threshold `0.35`.

| Threshold | Số prediction của sample `kitchen` |
| --- | ---: |
| `0.20` | 17 |
| `0.35` | 11 |
| `0.60` | 6 |

Khi hạ threshold từ `0.35` xuống `0.20`, số prediction tăng từ 11 lên 17, thêm 6 prediction gồm 3 `spoon`, 1 `potted plant`, 1 `dining table` và 1 `bottle`. Reviewer cần kiểm tra thêm các prediction này về lớp, vị trí box và khả năng dự đoán sai. Ngưỡng thấp có thể giúp tìm thêm đối tượng nhưng không bảo đảm các prediction bổ sung đều đúng.

Khi tăng threshold từ `0.35` lên `0.60`, số prediction giảm từ 11 xuống 6, loại 3 prediction `bowl` và 2 `cup`; còn lại 2 `person`, 2 `bowl` và 2 `oven`. Số prediction cần kiểm tra giảm, nhưng các đối tượng có điểm thấp có thể bị bỏ sót. Threshold chỉ lọc prediction, không phải quy tắc bỏ qua đối tượng khi tạo ground truth. Chưa có ground truth đối chiếu nên không thể kết luận precision hoặc recall thực tế từ các số lượng này.

**Quy tắc box chặt đề xuất:** Với quy ước gán nhãn phần nhìn thấy, vẽ hình chữ nhật nhỏ nhất bao hết phần nhìn thấy của từng đối tượng, hạn chế nền thừa và không cắt mất phần đối tượng có thể quan sát. Mỗi đối tượng có một box riêng.

Với đối tượng bị che khuất hoặc cắt mép, guideline cần xác định vẽ theo phần nhìn thấy hay ước lượng toàn bộ đối tượng, mức độ nhìn thấy tối thiểu để gán nhãn và cách đánh dấu trường hợp che khuất/cắt mép. Nếu chưa có quy định hoặc không đủ căn cứ xác định lớp, cần chuyển reviewer hoặc người phụ trách guideline quyết định. Ảnh có một prediction `person` ở mép trái chỉ bao vùng cơ thể nhìn thấy một phần; đây là trường hợp cần kiểm tra theo quy tắc đó.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn bằng chứng: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

**Một record, trích năm điểm đầu của polygon:**

```json
{
  "instance_id": "kitchen-001",
  "class_name": "person",
  "score": 0.899318,
  "polygon_point_count": 348,
  "polygon_xy_excerpt": [
    [446.0, 70.0],
    [445.0, 71.0],
    [444.0, 71.0],
    [443.0, 72.0],
    [442.0, 72.0]
  ]
}
```

`polygon_xy_excerpt` ở trên là phần trích minh họa từ trường `polygon_xy`, không phải tên trường trong file nguồn. Tọa độ dùng đơn vị pixel của ảnh gốc. Sample `kitchen` có 11 instance trong file segmentation.

Polygon mô tả đường biên của từng instance, bổ sung hình dạng chi tiết so với box chữ nhật. Ví dụ, mask người bám theo đầu, vai, tay và chân, giúp tách phần cơ thể khỏi phần nền nằm trong box, như khoảng trống giữa hai chân.

`instance_id` dùng để phân biệt và tham chiếu từng đối tượng trong đầu ra, kể cả khi nhiều đối tượng cùng lớp. `kitchen-001` không phải ID lớp, không phải danh tính cá nhân và không tự động là ID theo dõi ổn định của cùng một người qua nhiều ảnh hoặc video.

**Quy tắc biên mask đề xuất:** Bám theo biên phần đối tượng nhìn thấy, không tô lan sang nền hoặc vật thể khác và giữ riêng các instance tiếp xúc nhau. Không tự suy đoán phần bị che khuất nếu guideline yêu cầu chỉ gán phần nhìn thấy.

Với vùng mờ, tiếp xúc hoặc che khuất, guideline cần quy định cách chọn biên, xử lý phần bị che, vùng rời nhau và lỗ bên trong mask. Nếu không thể xác định nhất quán, annotator đánh dấu vùng mơ hồ và chuyển reviewer quyết định. Trong ảnh, các bát ở mép trái tiếp xúc hoặc che nhau, còn mask bàn trải rộng qua vùng có nhiều vật dụng; cần đối chiếu kỹ biên giữa bàn và đồ vật. Các nhãn dụng cụ treo ở góc phải cũng chồng lên nhau, gây khó đọc khi kiểm tra bằng ảnh tổng quan.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Ảnh thô được gán nhãn theo guideline và kiểm tra để tạo ground truth. Ground truth có thể dùng cho huấn luyện và đánh giá; mô hình tạo prediction trên ảnh đầu vào. QC đối chiếu prediction với ảnh và quy tắc gán nhãn, xác định mục cần sửa hoặc chuyển xử lý. Các file hiện có là prediction của checkpoint, không phải ground truth đã được duyệt và không phải bằng chứng bài thực hành đã huấn luyện mô hình mới.

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cấp ảnh hoặc tập nhãn theo guideline; lưu ID, tên lớp và taxonomy | Ảnh `traffic` có nhiều loại phương tiện nhưng top-1 là `cab`; một nhãn không mô tả đầy đủ cảnh | Áp dụng quy tắc chọn chủ thể chính hoặc đa nhãn; đánh dấu khi mơ hồ | Nhãn có phù hợp nội dung và taxonomy không; có dùng score thay cho xác minh không |
| Phát hiện vật thể | Mỗi đối tượng có lớp và box, với định dạng và đơn vị tọa độ rõ ràng | Người ở mép trái chỉ nhìn thấy một phần; nhiều box của bát/cốc tập trung và chồng nhau | Kiểm tra từng đối tượng, sửa lớp và box theo quy tắc phần nhìn thấy/che khuất | Đối tượng bỏ sót, box trùng, độ chặt, sai lớp và tính nhất quán tại mép ảnh |
| Instance segmentation | Mỗi instance có lớp và mask/polygon riêng | Biên bát và bàn khó tách ở vùng tiếp xúc/che khuất; nhãn dụng cụ treo chồng nhau trên ảnh minh họa | Kiểm tra biên từng mask, tách instance và đánh dấu vùng chưa rõ | Mask có lấn nền hoặc vật khác, thiếu phần nhìn thấy, gộp/tách sai instance hay không |

Các điểm nêu trên là vị trí cần QC theo ảnh minh họa, không phải kết luận lỗi định lượng đã được xác minh bằng ground truth.

## 5. An toàn dữ liệu

**Quy tắc bảo vệ dữ liệu:** Chỉ sử dụng ảnh được cấp phép và thuộc phạm vi bài thực hành; không đưa họ tên, MSSV, email, số điện thoại hoặc dữ liệu nhạy cảm của người học vào báo cáo và output. Giữ thông tin ghi công nguồn ảnh theo tài liệu attribution.

Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng xử lý và báo cho giảng viên hoặc người phụ trách bộ dữ liệu qua kênh được quy định, không tiếp tục phát tán dữ liệu. Đầu mối và kênh liên hệ cụ thể chưa được cung cấp.

Theo `IMAGE_ATTRIBUTION.md`, các ảnh được tải từ COCO 2017 validation:

| Sample | Tác phẩm / tác giả | Nguồn được ghi trong attribution | Giấy phép được ghi |
| --- | --- | --- | --- |
| `traffic` — COCO 210273 | Wuhan / Tauno Tõhk (toehk) | [Flickr](https://www.flickr.com/photo.gne?id=5336041838) | CC BY 2.0 |
| `kitchen` — COCO 397133 | Kitchen / Maggie Stephens (Pot Noodle) | [Flickr](https://www.flickr.com/photo.gne?id=6255196340) | CC BY 2.0 |
| `dining` — COCO 166918 | Big Wine Thing / WordRidden | [Flickr](https://www.flickr.com/photo.gne?id=4745624149) | CC BY 2.0 |

Các PNG bằng chứng là phiên bản đã thêm lớp phủ prediction hoặc biểu đồ. Thông tin nguồn và giấy phép trong bảng được trích từ file attribution đã cung cấp, chưa kiểm chứng độc lập tại trang nguồn.

## 6. Danh sách bằng chứng

- [x] `classification_predictions.json` — đã đọc.
- [x] `detection_predictions.json` — đã đọc.
- [x] `segmentation_predictions.json` — đã đọc.
- [x] `IMAGE_ATTRIBUTION.md` — đã đọc.
- [x] `visuals/classification_top5.png` — đã xem ảnh cùng tên được cung cấp.
- [x] `visuals/detection_predictions.png` — đã xem ảnh cùng tên được cung cấp.
- [x] `visuals/segmentation_prediction.png` — đã xem ảnh cùng tên được cung cấp.
- [ ] Ô validation cuối notebook báo `PASS` — chưa có output để xác nhận.
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong toàn bộ báo cáo/output — báo cáo không đưa thông tin định danh người học; chưa xác nhận kiểm tra đầy đủ toàn bộ output/notebook. Tên tác giả ảnh được giữ để ghi công nguồn.

**Các thông tin còn cần bổ sung:** ngày chạy, CPU/GPU, phiên bản Python/PyTorch, thay đổi so với notebook nguồn và output validation cuối notebook. Các đường dẫn `visuals/` ở trên được tính từ thư mục `day1_lab_outputs/` chứa bằng chứng.
