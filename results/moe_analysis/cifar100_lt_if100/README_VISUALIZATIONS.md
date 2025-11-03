# Hướng Dẫn Hiển Thị Visualizations Trong Markdown Report

## Cách Hiển Thị Ảnh Trong Markdown

Report `MOE_L2R_Analysis_Report.md` đã được cấu hình để hiển thị các plots trực tiếp bằng cú pháp:

```markdown
![Alt Text](filename.png)

*Figure N: Description*
```

## Các Plot Đã Được Tích Hợp

Tất cả các plots sau đã được thêm vào report:

1. ✅ `gating_statistics.png` - Figure 1
2. ✅ `gating_weight_distribution.png` - Figure 2
3. ✅ `oracle_comparison.png` - Figure 3
4. ✅ `nll_decomposition.png` - Figure 4
5. ✅ `calibration_vs_coverage_*.png` - Figures 5a-5b
6. ✅ `variance_analysis.png` - Figure 6
7. ✅ `mi_analysis.png` - Figure 7
8. ✅ `disagreement_analysis.png` - Figure 8
9. ✅ `shuffle_gating.png` - Figure 10
10. ✅ `margin_analysis.png` - Figure 11
11. ✅ `rejection_heatmap.png` - Figure 12
12. ✅ `alpha_coverage.png` - Figure 13
13. ✅ `calibration_*.png` - Figures 14a-14e
14. ✅ `variance_nll_correlation.png` - Figure 15
15. ✅ `oracle_uniform_gating_rc.png` - Figure 16

## Plot Nằm Ở Thư Mục Khác

Một số plots từ L2R plugin nằm ở thư mục khác:

- `ltr_rc_curves_balanced_gating_test.png` - Nằm ở `results/ltr_plugin/cifar100_lt_if100/`

Để hiển thị plots từ thư mục khác, có thể:

1. **Copy file** vào thư mục `results/moe_analysis/cifar100_lt_if100/`
2. **Dùng relative path** trong markdown:
   ```markdown
   ![RC Curves](../ltr_plugin/cifar100_lt_if100/ltr_rc_curves_balanced_gating_test.png)
   ```
3. **Dùng absolute path** (nếu viewer hỗ trợ)

## Xem Report Với Visualizations

### Trong VS Code / Cursor:
1. Mở file `MOE_L2R_Analysis_Report.md`
2. Preview markdown (Ctrl+Shift+V hoặc click Preview icon)
3. Các plots sẽ hiển thị tự động nếu nằm cùng thư mục

### Chuyển Sang PDF:
Có thể dùng pandoc hoặc markdown-to-pdf tool:

```bash
# Với pandoc (cần cài đặt)
pandoc MOE_L2R_Analysis_Report.md -o report.pdf --pdf-engine=xelatex

# Hoặc dùng VS Code extension "Markdown PDF"
```

### Chuyển Sang HTML:
```bash
pandoc MOE_L2R_Analysis_Report.md -o report.html --standalone
```

## Lưu Ý

- Tất cả plots phải nằm trong cùng thư mục với markdown file
- Hoặc dùng relative/absolute paths phù hợp
- Đảm bảo tên file chính xác (case-sensitive)

