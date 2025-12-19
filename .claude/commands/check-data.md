# Check Training Data

Analyze the BTC.CSV training data file:

1. Check if BTC.CSV exists and its size
2. Preview the first few rows of data
3. Report on data quality (missing values, date range, OHLCV columns)
4. Suggest any data improvements needed

```bash
if [ -f "BTC.CSV" ]; then
    echo "File size: $(ls -lh BTC.CSV | awk '{print $5}')"
    echo "Row count: $(wc -l < BTC.CSV)"
    head -5 BTC.CSV
else
    echo "BTC.CSV not found"
fi
```
