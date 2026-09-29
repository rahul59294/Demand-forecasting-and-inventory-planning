# Data Audit Report: UCI Online Retail II Dataset

**Dataset Source**: UCI Machine Learning Repository (*Online Retail II*)
**Audit Status**: STRICTLY UNCLEANED (0 rows dropped or modified)
**Total Records**: 1,067,371 rows | **Total Columns**: 8
**Date Span**: 2009-12-01 07:45:00 to 2011-12-09 12:50:00 (738 days)

---

## 1. Dataset Overview & Schema
- **Shape**: `1,067,371` rows × `8` columns
- **Date Range**: `2009-12-01 07:45:00` to `2011-12-09 12:50:00`
- **Total Time Window**: `738` days (approx. 2 full calendar years)

| Column Name | Inferred Dtype | Sample Value |
| :--- | :--- | :--- |
| `Invoice` | `str` | 489434 |
| `StockCode` | `str` | 85048 |
| `Description` | `str` | 15CM CHRISTMAS GLASS BALL 20 LIGHTS |
| `Quantity` | `int64` | 12 |
| `InvoiceDate` | `datetime64[us]` | 2009-12-01 07:45:00 |
| `Price` | `float64` | 6.95 |
| `Customer ID` | `float64` | 13085.0 |
| `Country` | `str` | United Kingdom |

---

## 2. Missing Values Analysis
| Column Name | Missing Count | Missing Percentage |
| :--- | :---: | :---: |
| `Invoice` | 0 | 0.00% |
| `StockCode` | 0 | 0.00% |
| `Description` | 4,382 | 0.41% |
| `Quantity` | 0 | 0.00% |
| `InvoiceDate` | 0 | 0.00% |
| `Price` | 0 | 0.00% |
| `Customer ID` | 243,007 | 22.77% |
| `Country` | 0 | 0.00% |

> **Key Observation**: `Customer ID` is missing in **243,007 rows (22.77%)**, which corresponds to guest checkouts, marketplace/Amazon bulk orders, or non-account transactions. `Description` is missing in **4,382 rows (0.41%)**.

---

## 3. Duplicate Rows
- **Exact Duplicate Rows**: `34,335` rows (`3.22%` of the dataset)
- **Nature of Duplicates**: Repeated line items across identical `Invoice`, `StockCode`, `Customer ID`, and `Price`. In retail logging, these can occur from system retries, double-clicks, or multi-box shipments.

---

## 4. Cancelled Invoices
- **Cancelled Invoices Count**: `19,494` rows
- **Share of Total Rows**: `1.83%`
- **Cancellation & Quantity Interaction**: Of the 19,494 cancelled transactions, `19,493` have negative quantities (`Quantity < 0`). However, `3,457` rows have negative quantities with *normal* invoice numbers (e.g. inventory adjustments, damages, write-offs).

---

## 5. Non-Positive Quantities and Prices
| Metric | Count | % of Total Rows | Notes |
| :--- | :---: | :---: | :--- |
| **Quantity <= 0** | 22,950 | 2.15% | Cancellations, returns, damages, write-offs |
| — Quantity < 0 | 22,950 | 2.15% | Returns, adjustments, broken goods |
| — Quantity == 0 | 0 | 0.00% | Administrative placeholders |
| **Price <= 0** | 6,207 | 0.58% | Free items, promotional gifts, system adjustments |
| — Price == 0 | 6,202 | 0.58% | Zero-price transactions |
| — Price < 0 | 5 | 0.0005% | Accounting adjustments / bad debts |
| **Both Quantity <= 0 & Price <= 0** | 3,457 | 0.32% | Damaged stock removals / ledger corrections |

---

## 6. Non-Product StockCodes
Typical retail items use a 5-digit number with an optional 1-3 letter color/size suffix (`^\d{5}[A-Za-z]{0,3}$`).
- **Standard Product Rows**: `1,061,278` (`99.43%`)
- **Non-Standard / Fee / Service Rows**: `6,093` (`0.57%`)
- **Unique Non-Product Codes Detected**: `62` distinct codes

### Breakdown of Most Frequent Non-Product StockCodes
| StockCode | Description / Type | Row Count | % of Non-Product Rows |
| :--- | :--- | :---: | :---: |
| `POST` | POSTAGE | 2,122 | 34.83% |
| `DOT` | DOTCOM POSTAGE | 1,446 | 23.73% |
| `M` | Manual | 1,421 | 23.32% |
| `C2` | CARRIAGE | 282 | 4.63% |
| `D` | Discount | 177 | 2.90% |
| `S` | SAMPLES | 104 | 1.71% |
| `BANK CHARGES` | Bank Charges | 102 | 1.67% |
| `ADJUST` | Adjustment by john on 26/01/2010 16 | 67 | 1.10% |
| `AMAZONFEE` | AMAZON FEE | 43 | 0.71% |
| `DCGS0058` | MISO PRETTY  GUM | 31 | 0.51% |
| `gift_0001_20` | Dotcomgiftshop Gift Voucher £20.00 | 29 | 0.48% |
| `gift_0001_30` | Dotcomgiftshop Gift Voucher £30.00 | 29 | 0.48% |
| `DCGSSGIRL` | update | 25 | 0.41% |
| `DCGSSBOY` | update | 23 | 0.38% |
| `PADS` | PADS TO MATCH ALL CUSHIONS | 19 | 0.31% |
| `gift_0001_10` | Dotcomgiftshop Gift Voucher £10.00 | 16 | 0.26% |
| `CRUK` | CRUK Commission | 16 | 0.26% |
| `DCGS0076` | SUNJAR LED NIGHT NIGHT LIGHT | 15 | 0.25% |
| `TEST001` | This is a test product. | 15 | 0.25% |
| `DCGS0003` | BOXED GLASS ASHTRAY | 14 | 0.23% |
| `gift_0001_50` | Dotcomgiftshop Gift Voucher £50.00 | 8 | 0.13% |
| `gift_0001_40` | Dotcomgiftshop Gift Voucher £40.00 | 7 | 0.11% |
| `DCGS0069` | OOH LA LA DOGS COLLAR | 6 | 0.10% |
| `B` | Adjust bad debt | 6 | 0.10% |
| `DCGS0004` | HAYNES CAMPER SHOULDER BAG | 5 | 0.08% |

---

## 7. Unique Entity Counts
| Entity Type | Count | Context / Remarks |
| :--- | :---: | :--- |
| **Unique StockCodes (SKUs)** | `5,305` | Total unique codes appearing in raw dataset |
| — Standard Physical SKUs | `5,243` | Standard 5-digit catalog items |
| — Non-Product / Service Codes | `62` | Post, manual, fees, tests, adjustments |
| **Unique Customers** | `5,942` | Distinct Customer IDs (excludes `243,007` null rows) |
| **Unique Invoices** | `53,628` | Total distinct invoice numbers (including cancellations) |
| **Unique Countries** | `43` | Distinct country names represented |

---

## 8. Top 10 Countries by Revenue
*Revenue computed on raw data as `Quantity * Price` (uncleaned baseline).*

| Rank | Country | Invoices | Units Sold | Total Revenue (Raw) | Revenue Share |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | **United Kingdom** | 49,108 | 8,692,875 | £16,382,583.90 | 84.94% |
| 2 | **EIRE** | 806 | 331,341 | £615,519.55 | 3.19% |
| 3 | **Netherlands** | 250 | 381,951 | £548,524.95 | 2.84% |
| 4 | **Germany** | 1,095 | 224,581 | £417,988.56 | 2.17% |
| 5 | **France** | 746 | 184,952 | £328,191.80 | 1.70% |
| 6 | **Australia** | 117 | 103,706 | £167,129.07 | 0.87% |
| 7 | **Switzerland** | 123 | 52,378 | £99,728.76 | 0.52% |
| 8 | **Spain** | 188 | 45,156 | £91,859.48 | 0.48% |
| 9 | **Sweden** | 129 | 87,875 | £87,809.42 | 0.46% |
| 10 | **Denmark** | 53 | 235,218 | £65,741.09 | 0.34% |

> **Insight**: The **United Kingdom** accounts for **84.9%** of all raw recorded revenue, followed by EIRE, Netherlands, Germany, and France.

---

## 9. SKUs Mapping to Multiple Descriptions
- **SKUs with Multiple Descriptions**: `1,212` SKUs (`22.85%` of all SKUs)
- **Root Causes**: Minor typos, casing differences (`WHITE CHERRY LIGHTS` vs ` WHITE CHERRY LIGHTS`), packaging updates, or codes reused for different catalog variants.

### Top 10 SKUs with Most Distinct Descriptions
| StockCode | Distinct Descriptions | Sample Descriptions |
| :--- | :---: | :--- |
| `20713` | 8 | JUMBO BAG OWLS | MISSING | WRONGLY MARKED. 23343 IN BOX | WRONGLY CODED-23343 |
| `21181` | 7 | PLEASE ONE PERSON  METAL SIGN | PLEASE ONE PERSON METAL SIGN | MISSING | ON CARGO ORDER |
| `22423` | 7 | REGENCY CAKESTAND 3 TIER | SMASHED | DAMAGED | BROKEN, UNEVEN BOTTOM |
| `22734` | 7 | SET OF 6 RIBBONS VINTAGE CHRISTMAS | CARTON QNTY WAS 216 NOT 144 AS STAT | AMAZON ADJUSTMENT | AMENDMENT |
| `23084` | 7 | RABBIT NIGHT LIGHT | TEMP ADJUSTMENT | ALLOCATE STOCK FOR DOTCOM ORDERS TA | ADD STOCK TO ALLOCATE ONLINE ORDERS |
| `22719` | 6 | GUMBALL COATHOOK., BLACK & WHITE | GUMBALL MONOCHROME COAT RACK | 22467 | WRONG BARCODE (22467) |
| `21830` | 6 | ASSORTED CREEPY CRAWLIES | MERCHANT CHANDLER CREDIT ERROR, STO | SOLD AS 1 | ? |
| `85175` | 6 | CACTI T-LIGHT CANDLES | DOTCOM SOLD SETS | AMAZON SOLD SETS | WRONGLY SOLD SETS |
| `47566B` | 6 | TEA TIME PARTY BUNTING | MISSING | CORRECT PREVIOUS ADJUSTMENT | STOCK CREDITED FROM ROYAL YACHT INC |
| `22740` | 5 | POLKA DOT PEN | POLKADOT PEN | POLKADOT PENS | ? |

---

## 10. Outliers Analysis
### Top 10 Largest Positive Quantities
| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `581483` | `23843` | PAPER CRAFT , LITTLE BIRD | **80,995** | £2.08 | 2011-12-09 | 16446 | United Kingdom |
| `541431` | `23166` | MEDIUM CERAMIC TOP STORAG | **74,215** | £1.04 | 2011-01-18 | 12346 | United Kingdom |
| `497946` | `37410` | BLACK AND WHITE PAISLEY F | **19,152** | £0.10 | 2010-02-15 | 13902 | Denmark |
| `501534` | `21099` | SET/6 STRAWBERRY PAPER CU | **12,960** | £0.10 | 2010-03-17 | 13902 | Denmark |
| `501534` | `21091` | SET/6 WOODLAND PAPER PLAT | **12,960** | £0.10 | 2010-03-17 | 13902 | Denmark |
| `501534` | `21085` | SET/6 WOODLAND PAPER CUPS | **12,744** | £0.10 | 2010-03-17 | 13902 | Denmark |
| `578841` | `84826` | ASSTD DESIGN 3D PAPER STI | **12,540** | £0.00 | 2011-11-25 | 13256 | United Kingdom |
| `501534` | `21092` | SET/6 STRAWBERRY PAPER PL | **12,480** | £0.10 | 2010-03-17 | 13902 | Denmark |
| `507637` | `84016` | FLAG OF ST GEORGE CAR FLA | **10,200** | £0.00 | 2010-05-10 | Guest | United Kingdom |
| `502269` | `21984` | PACK OF 12 PINK PAISLEY T | **10,000** | £0.25 | 2010-03-23 | 17940 | United Kingdom |

### Top 10 Largest Negative Quantities (Massive Returns/Adjustments)
| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `C581484` | `23843` | PAPER CRAFT , LITTLE BIRD | **-80,995** | £2.08 | 2011-12-09 | 16446 | United Kingdom |
| `C541433` | `23166` | MEDIUM CERAMIC TOP STORAG | **-74,215** | £1.04 | 2011-01-18 | 12346 | United Kingdom |
| `519017` | `22759` | nan | **-9,600** | £0.00 | 2010-08-13 | Guest | United Kingdom |
| `556690` | `23005` | printing smudges/thrown a | **-9,600** | £0.00 | 2011-06-14 | Guest | United Kingdom |
| `556691` | `23005` | printing smudges/thrown a | **-9,600** | £0.00 | 2011-06-14 | Guest | United Kingdom |
| `C536757` | `84347` | ROTATING SILVER ANGELS T- | **-9,360** | £0.03 | 2010-12-02 | 15838 | United Kingdom |
| `C536757` | `84347` | ROTATING SILVER ANGELS T- | **-9,360** | £0.03 | 2010-12-02 | 15838 | United Kingdom |
| `504311` | `22197` | nan | **-9,200** | £0.00 | 2010-04-12 | Guest | United Kingdom |
| `556687` | `23003` | Printing smudges/thrown a | **-9,058** | £0.00 | 2011-06-14 | Guest | United Kingdom |
| `507913` | `10120` | Zebra invcing error | **-9,000** | £0.00 | 2010-05-11 | Guest | United Kingdom |

### Top 10 Highest Unit Prices
| Invoice | StockCode | Description | Quantity | Price | Date | Customer ID | Country |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `C556445` | `M` | Manual | -1 | **£38,970.00** | 2011-06-10 | 15098 | United Kingdom |
| `C512770` | `M` | Manual | -1 | **£25,111.09** | 2010-06-17 | 17399 | United Kingdom |
| `512771` | `M` | Manual | 1 | **£25,111.09** | 2010-06-17 | Guest | United Kingdom |
| `C520667` | `BANK CHARGES` | Bank Charges | -1 | **£18,910.69** | 2010-08-27 | Guest | United Kingdom |
| `C580605` | `AMAZONFEE` | AMAZON FEE | -1 | **£17,836.46** | 2011-12-05 | Guest | United Kingdom |
| `C540117` | `AMAZONFEE` | AMAZON FEE | -1 | **£16,888.02** | 2011-01-05 | Guest | United Kingdom |
| `C540118` | `AMAZONFEE` | AMAZON FEE | -1 | **£16,453.71** | 2011-01-05 | Guest | United Kingdom |
| `C537630` | `AMAZONFEE` | AMAZON FEE | -1 | **£13,541.33** | 2010-12-07 | Guest | United Kingdom |
| `537632` | `AMAZONFEE` | AMAZON FEE | 1 | **£13,541.33** | 2010-12-07 | Guest | United Kingdom |
| `C537651` | `AMAZONFEE` | AMAZON FEE | -1 | **£13,541.33** | 2010-12-07 | Guest | United Kingdom |

---

## 11. Decisions Required Before Cleaning (Next Steps)
Before starting feature engineering, demand forecasting, and inventory optimization, the following strategic decisions must be finalized:
1. **Cancelled Orders & Returns (`Invoice` starting with 'C' & `Quantity < 0`)**:
   - *Option A*: Match and deduct cancellations against original orders to reflect net demand.
   - *Option B*: Filter out cancellations completely when forecasting gross demand / sales velocity.
2. **Non-Product StockCodes (`POST`, `DOT`, `M`, `BANK CHARGES`, `AMAZONFEE`, etc.)**:
   - These are postage, carrier fees, manual adjustments, and internal debt entries. For physical inventory replenishment and demand forecasting, these should be excluded so SKU-level forecasts reflect physical product demand.
3. **Missing `Customer ID` (22.77% of rows)**:
   - Essential for customer-level cohort or LTV models, but for SKU-level demand forecasting and aggregate inventory safety stock, these represent valid transaction volume and should generally be retained (unless unverified).
4. **Extreme Quantity Outliers (e.g. +80,995 and -80,995 units on SKU 23843)**:
   - These are single-customer wholesale bulk orders or administrative inventory clearing events that will distort time-series models unless winsorized, isolated, or flagged.
5. **Price = 0 Transactions (6,207 rows)**:
   - Many are damaged goods write-offs (`lost`, `thrown away`, `damaged`). These should be filtered out from sales demand or redirected to shrinkage analytics.