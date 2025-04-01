import pandas as pd
import difflib
import re
import unicodedata
import numpy as np


def is_kanji(char):
    """
    Check if a character is a kanji (CJK Unified Ideograph)
    """
    return 'CJK UNIFIED IDEOGRAPH' in unicodedata.name(char, '')


def contains_kanji(text):
    """
    Check if text contains any kanji characters
    """
    if pd.isna(text):
        return False

    text = str(text)
    return any(is_kanji(char) for char in text)


def has_numeric_fullwidth_to_halfwidth(original, modified):
    """
    Check if numbers changed from full-width to half-width
    """
    if pd.isna(original) or pd.isna(modified):
        return False

    original_str = str(original)
    modified_str = str(modified)

    # Create regex pattern to find full-width digits in original
    fw_digits_pattern = re.compile(r'[０-９]')

    # Dictionary mapping full-width to half-width digits
    fw_to_hw_digits = {
        '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
        '５': '5', '６': '6', '７': '7', '８': '8', '９': '9'
    }

    # Find all full-width digits in original
    fw_digits_matches = fw_digits_pattern.finditer(original_str)

    for match in fw_digits_matches:
        idx = match.start()
        fw_digit = original_str[idx]

        # Check if there's a corresponding position in modified text with the half-width equivalent
        if idx < len(modified_str):
            # Need to handle cases where the modified text might have katakana replacing some digits
            if modified_str[idx] == fw_to_hw_digits.get(fw_digit):
                return True
            # Look ahead or behind in case there's position shift due to replacements
            for offset in range(-3, 4):  # Check a few positions before and after
                pos = idx + offset
                if 0 <= pos < len(modified_str) and modified_str[pos] == fw_to_hw_digits.get(fw_digit):
                    return True

    # If we find full-width in original and half-width in modified text at similar positions
    d = difflib.SequenceMatcher(None, original_str, modified_str)
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag == 'replace':
            orig_chars = original_str[i1:i2]
            mod_chars = modified_str[j1:j2]

            # Check if this replacement includes full-width to half-width digit conversion
            for c1 in orig_chars:
                if c1 in fw_to_hw_digits and fw_to_hw_digits[c1] in mod_chars:
                    return True

    return False


def has_symbol_to_kana_conversion(original, modified):
    """
    Check if symbols in original are replaced with kana in modified
    Examples:
    - "６－２－２－７３４" -> "6メ2-2７３４" (dash replaced with メ)
    - Half-width dash (-) to kana
    - Full-width dash (－) to kana
    """
    if pd.isna(original) or pd.isna(modified):
        return False

    original_str = str(original)
    modified_str = str(modified)

    # Create regex patterns for common symbols - both half-width and full-width
    symbols_pattern = re.compile(r'[－-／/・]')  # Includes both width variants of dash, slash, dot

    # Dictionary mapping common symbols to possible kana replacements
    symbol_to_kana_map = {
        '-': ['メ', 'ノ'],  # Half-width dash potential replacements
        '－': ['メ', 'ノ'],  # Full-width dash potential replacements
        '/': ['ノ'],  # Half-width slash potential replacements
        '／': ['ノ'],  # Full-width slash potential replacements
        '・': ['ノ'],  # Middle dot potential replacements
    }

    # Find all symbols in original
    symbol_matches = symbols_pattern.finditer(original_str)

    for match in symbol_matches:
        idx = match.start()
        symbol = original_str[idx]

        # Potential kana replacements for this symbol
        potential_kana = symbol_to_kana_map.get(symbol, [])

        # Check if this position in modified has a kana character
        if idx < len(modified_str):
            # Direct position check
            if idx < len(modified_str) and re.match(r'[\u3040-\u309F\u30A0-\u30FF]', modified_str[idx]):
                # If it's one of our expected kana replacements, it's very likely a conversion
                if modified_str[idx] in potential_kana:
                    return True
                # Otherwise, it's still possibly a conversion
                return True

            # Check nearby positions in case of alignment shifts
            for offset in range(-3, 4):
                pos = idx + offset
                if 0 <= pos < len(modified_str):
                    if re.match(r'[\u3040-\u309F\u30A0-\u30FF]', modified_str[pos]):
                        # If it's one of our expected kana replacements, it's very likely a conversion
                        if modified_str[pos] in potential_kana:
                            return True

    # Look at the diff to find symbol-to-kana replacements
    d = difflib.SequenceMatcher(None, original_str, modified_str)
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag == 'replace':
            orig_chars = original_str[i1:i2]
            mod_chars = modified_str[j1:j2]

            # Check if original contains symbols and modified contains kana
            if re.search(r'[－-／/・]', orig_chars) and re.search(r'[\u3040-\u309F\u30A0-\u30FF]', mod_chars):
                # Specific case check: any dash replaced with メ or ノ
                dash_symbols = ['-', '－']
                slash_symbols = ['/', '／']
                dot_symbols = ['・']

                if any(sym in orig_chars for sym in dash_symbols) and any(k in mod_chars for k in ['メ', 'ノ']):
                    return True
                if any(sym in orig_chars for sym in slash_symbols) and 'ノ' in mod_chars:
                    return True
                if any(sym in orig_chars for sym in dot_symbols) and 'ノ' in mod_chars:
                    return True
                # Generic case
                return True

    return False


def compare_addresses(original, modified):
    """
    Compare two address strings and identify the specific differences.
    Works with Japanese characters (kanji, katakana, hiragana, etc.)
    Also detects subtle encoding differences.

    Args:
        original (str): Original address (住所カナ1)
        modified (str): Modified address (住所カナ2)

    Returns:
        str: Description of the differences
    """
    # Handle None/NaN values
    if pd.isna(original) and pd.isna(modified):
        return "No Change"
    if pd.isna(original):
        return "Added: entire address"
    if pd.isna(modified):
        return "Removed: entire address"

    # Convert to string if they're not already
    original = str(original) if not pd.isna(original) else ""
    modified = str(modified) if not pd.isna(modified) else ""

    # Check if visually identical but with different encodings
    if original == modified:
        return "No Change"
    elif original.strip() == modified.strip():
        return "Whitespace difference (leading/trailing spaces)"

    # Check for half-width vs full-width character differences
    def is_only_width_different(str1, str2):
        # Maps for half-width to full-width conversions and vice versa
        hw_to_fw = {
            '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
            '5': '５', '6': '６', '7': '７', '8': '８', '9': '９',
            '-': 'ー', '.': '．', '(': '（', ')': '）', ' ': '　',
            '/': '／', ',': '，', ':': '：'
        }

        fw_to_hw = {v: k for k, v in hw_to_fw.items()}

        # If lengths are different, can't be just width difference
        if len(str1) != len(str2):
            return False

        for c1, c2 in zip(str1, str2):
            # If characters are the same, continue
            if c1 == c2:
                continue

            # Check if it's a width conversion (either direction)
            if hw_to_fw.get(c1) == c2 or fw_to_hw.get(c1) == c2:
                continue

            # If we get here, found a difference that isn't just width
            return False

        # If we get through the whole string, it's only width differences
        return True

    # Check for unicode normalization differences
    if unicodedata.normalize('NFKC', original) == unicodedata.normalize('NFKC', modified):
        return "Unicode normalization difference"

    # Check for symbol to kana conversion
    if has_symbol_to_kana_conversion(original, modified):
        return "Symbol to kana conversion (e.g., dash to メ)"

    # Check for full-width to half-width numeric conversion
    if has_numeric_fullwidth_to_halfwidth(original, modified):
        return "Full-width to half-width numeric conversion"

    try:
        # Generate detailed diff using SequenceMatcher
        d = difflib.SequenceMatcher(None, original, modified)
        result = []

        # Process each operation code in the diff
        for tag, i1, i2, j1, j2 in d.get_opcodes():
            if tag == 'replace':
                # Get character codes for better debugging
                orig_chars = original[i1:i2]
                mod_chars = modified[j1:j2]
                orig_codes = [f"{c}({ord(c)})" for c in orig_chars]
                mod_codes = [f"{c}({ord(c)})" for c in mod_chars]

                result.append(f"Changed: '{orig_chars}' → '{mod_chars}' [Codes: {orig_codes} → {mod_codes}]")
            elif tag == 'delete':
                orig_chars = original[i1:i2]
                orig_codes = [f"{c}({ord(c)})" for c in orig_chars]
                result.append(f"Removed: '{orig_chars}' [Codes: {orig_codes}]")
            elif tag == 'insert':
                mod_chars = modified[j1:j2]
                mod_codes = [f"{c}({ord(c)})" for c in mod_chars]
                result.append(f"Added: '{mod_chars}' [Codes: {mod_codes}]")

        return "; ".join(result)
    except Exception as e:
        return f"Error comparing: {str(e)}"


def main():
    # File path - update this to your Excel file path
    file_path = r'C:\Users\s-li.du\Desktop\Japanese Address.xlsx'
    output_path = r'C:\Users\s-li.du\Desktop\address_comparison_results1.xlsx'

    print(f"Loading data from {file_path}...")

    try:
        # Load the Excel file with your specific columns
        df = pd.read_excel(file_path)

        # Print column names to verify
        print("Column names in the file:", df.columns.tolist())

        # Make sure we're using the correct column names
        # Based on your updated column description
        no_column = 'No.'  # Column A
        kanji_column = '住所漢字'  # Column B
        kana1_column = '住所カナ1'  # Column C
        kana2_column = '住所カナ2'  # Column D

        # Check if these columns exist
        if kana1_column not in df.columns or kana2_column not in df.columns:
            print(f"ERROR: Could not find columns '{kana1_column}' or '{kana2_column}' in your file.")
            print("Available columns are:", df.columns.tolist())
            return

        print(f"Loaded {len(df)} rows. Comparing addresses...")

        # Create new columns to store results
        # First store the exact binary comparison (equivalent to Excel's EXACT)
        df['excel_exact'] = df.apply(
            lambda row: row[kana1_column] == row[kana2_column],
            axis=1
        )

        # Then store our string-based comparison (converts to string first)
        df['has_changed'] = df.apply(
            lambda row: str(row[kana1_column]) != str(row[kana2_column])
            if not pd.isna(row[kana1_column]) and not pd.isna(row[kana2_column])
            else pd.isna(row[kana1_column]) != pd.isna(row[kana2_column]),
            axis=1
        )

        # Count blank entries in 住所カナ2
        df['kana2_is_blank'] = df[kana2_column].isna()
        kana2_blank_count = df['kana2_is_blank'].sum()

        # Add new columns for kanji detection
        df['contains_kanji'] = df[kana2_column].apply(contains_kanji)

        # Add new column for numeric fullwidth-to-halfwidth conversion
        df['numeric_fullwidth_to_halfwidth'] = df.apply(
            lambda row: has_numeric_fullwidth_to_halfwidth(row[kana1_column], row[kana2_column]),
            axis=1
        )

        # Add new column for symbol-to-kana conversion
        df['symbol_to_kana'] = df.apply(
            lambda row: has_symbol_to_kana_conversion(row[kana1_column], row[kana2_column]),
            axis=1
        )

        # Add columns to check for 'チョウメ' and 'チョウ'
        df['contains_choume'] = df[kana2_column].apply(
            lambda x: False if pd.isna(x) else 'チョウメ' in str(x)
        )
        # Only count 'チョウ' if 'チョウメ' is not present
        df['contains_chou'] = df.apply(
            lambda row: False if pd.isna(row[kana2_column]) or row['contains_choume']
            else 'チョウ' in str(row[kana2_column]),
            axis=1
        )

        # Count how many are different according to Excel vs. Python
        excel_diff_count = (~df['excel_exact']).sum()
        python_diff_count = df['has_changed'].sum()
        kanji_count = df['contains_kanji'].sum()
        numeric_fw_to_hw_count = df['numeric_fullwidth_to_halfwidth'].sum()
        symbol_to_kana_count = df['symbol_to_kana'].sum()
        choume_count = df['contains_choume'].sum()
        chou_count = df['contains_chou'].sum()

        print(f"\nExcel EXACT() would find {excel_diff_count} differences")
        print(f"Python string comparison finds {python_diff_count} differences")
        print(f"Difference: {excel_diff_count - python_diff_count} rows have subtle encoding differences")
        print(f"Found {kana2_blank_count} rows where 住所カナ2 is blank")
        print(f"Found {kanji_count} rows where 住所カナ2 contains kanji")
        print(f"Found {numeric_fw_to_hw_count} rows where numbers changed from full-width to half-width")
        print(f"Found {symbol_to_kana_count} rows where symbols changed to kana characters")
        print(f"Found {choume_count} rows where 住所カナ2 contains 'チョウメ'")
        print(f"Found {chou_count} rows where 住所カナ2 contains 'チョウ' (excluding rows with 'チョウメ')")

        # Add column to identify difference between Excel and Python comparison
        df['excel_python_diff'] = ~df['excel_exact'] & ~df['has_changed']

        df['differences'] = df.apply(
            lambda row: compare_addresses(row[kana1_column], row[kana2_column]),
            axis=1
        )

        # Count changes for summary
        total_changes = df['has_changed'].sum()

        print(f"Analysis complete. Found {total_changes} changes out of {len(df)} addresses.")

        # Add detection for symbol width changes (half-width to full-width and vice versa)
        def has_symbol_width_change(original, modified):
            """
            Check if symbols changed width (half-width to full-width or vice versa)
            Example: "-" to "－" or "／" to "/"
            """
            if pd.isna(original) or pd.isna(modified):
                return False

            original_str = str(original)
            modified_str = str(modified)

            # Symbol mapping between half-width and full-width
            symbol_width_map = {
                '-': '－',  # Half-width to full-width dash
                '/': '／',  # Half-width to full-width slash
                '.': '．',  # Half-width to full-width period
                ',': '，',  # Half-width to full-width comma
                '(': '（',  # Half-width to full-width parenthesis
                ')': '）'  # Half-width to full-width parenthesis
            }
            # Add reverse mapping
            symbol_width_map.update({v: k for k, v in symbol_width_map.items()})

            # Look for direct symbol width changes
            for i, char in enumerate(original_str):
                if i < len(modified_str) and char in symbol_width_map and modified_str[i] == symbol_width_map[char]:
                    return True

            # Check the diff for width changes
            d = difflib.SequenceMatcher(None, original_str, modified_str)
            for tag, i1, i2, j1, j2 in d.get_opcodes():
                if tag == 'replace':
                    orig_chars = original_str[i1:i2]
                    mod_chars = modified_str[j1:j2]

                    for oc, mc in zip(orig_chars, mod_chars):
                        if oc in symbol_width_map and mc == symbol_width_map[oc]:
                            return True

            return False

        # Add the new detection to dataframe
        df['symbol_width_change'] = df.apply(
            lambda row: has_symbol_width_change(row[kana1_column], row[kana2_column]),
            axis=1
        )
        symbol_width_change_count = df['symbol_width_change'].sum()

        print(
            f"Found {symbol_width_change_count} rows where symbols changed width (half-width to full-width or vice versa)")

        # Initialize counters for different types of changes
        change_types = {
            'added': 0,  # New text added
            'removed': 0,  # Text removed
            'replaced': 0,  # Text replaced with something else
            'numeric_added': 0,  # Numbers added
            'symbols_added': 0,  # Symbols added (like -, —, etc.)
            'kana2_is_blank': kana2_blank_count,  # Number of rows where 住所カナ2 is blank
            'contains_kanji': kanji_count,  # Number of rows where 住所カナ2 contains kanji
            'numeric_fullwidth_to_halfwidth': numeric_fw_to_hw_count,  # Fullwidth to halfwidth numeric conversion
            'symbol_to_kana': symbol_to_kana_count,  # Symbol to kana conversion
            'symbol_width_change': symbol_width_change_count,  # Symbol width changes (half to full or vice versa)
            'contains_choume': choume_count,  # Number of rows where 住所カナ2 contains 'チョウメ'
            'contains_chou': chou_count  # Number of rows where 住所カナ2 contains 'チョウ' (excluding rows with 'チョウメ')
        }

        # Regular expressions for detecting specific types of changes
        import re

        # For all rows, analyze the type of change to catch subtle differences
        for idx, row in df.iterrows():
            # Get original values from both columns (before any conversion)
            original_val = row[kana1_column]
            modified_val = row[kana2_column]

            # Check if they're exactly equal in Python
            exact_match = original_val == modified_val

            # For rows that Python considers different
            if not exact_match:
                diff_text = row['differences']

                # Skip rows with errors
                if pd.isna(diff_text) or 'Error' in diff_text:
                    continue

                # Count substantive differences
                if 'Added:' in diff_text:
                    change_types['added'] += 1
                    # Check if added content contains numbers
                    if re.search(r'[0-9０-９]', diff_text):
                        change_types['numeric_added'] += 1
                    # Check if added content contains symbols
                    if re.search(r'[-－ー・／／]', diff_text):
                        change_types['symbols_added'] += 1

                elif 'Removed:' in diff_text:
                    change_types['removed'] += 1

                elif 'Changed:' in diff_text:
                    change_types['replaced'] += 1

        # Detailed analysis of rows containing kanji in 住所カナ2
        if kanji_count > 0:
            print(f"\nAnalyzing {kanji_count} rows where 住所カナ2 contains kanji...")

            # Sample a few rows with kanji
            kanji_rows = df[df['contains_kanji']].head(5)

            for idx, row in kanji_rows.iterrows():
                orig = row[kana1_column]
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ1: {orig}")
                print(f"カナ2: {mod}")

                # Identify kanji characters
                kanji_chars = [c for c in str(mod) if is_kanji(c)]
                print(f"Kanji characters found: {kanji_chars}")

        # Detailed analysis of full-width to half-width numeric conversions
        if numeric_fw_to_hw_count > 0:
            print(f"\nAnalyzing {numeric_fw_to_hw_count} rows where numbers changed from full-width to half-width...")

            # Sample a few rows with numeric conversions
            numeric_rows = df[df['numeric_fullwidth_to_halfwidth']].head(5)

            for idx, row in numeric_rows.iterrows():
                orig = row[kana1_column]
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ1: {orig}")
                print(f"カナ2: {mod}")

                # Find full-width digits in original
                fw_digits = re.findall(r'[０-９]', str(orig))
                # Find half-width digits in modified
                hw_digits = re.findall(r'[0-9]', str(mod))

                print(f"Full-width digits in カナ1: {fw_digits}")
                print(f"Half-width digits in カナ2: {hw_digits}")

        # Detailed analysis of symbol-to-kana conversions
        if symbol_to_kana_count > 0:
            print(f"\nAnalyzing {symbol_to_kana_count} rows where symbols changed to kana characters...")

            # Sample a few rows with symbol-to-kana conversions
            symbol_rows = df[df['symbol_to_kana']].head(5)

            for idx, row in symbol_rows.iterrows():
                orig = row[kana1_column]
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ1: {orig}")
                print(f"カナ2: {mod}")

                # Find symbols in original
                symbols = re.findall(r'[－-]', str(orig))
                # Find kana in modified that might have replaced symbols
                kana = re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', str(mod))

                print(f"Symbols in カナ1: {symbols}")
                print(f"Kana in カナ2 that might have replaced symbols: {kana}")

        # Create a summary dataframe with the requested changes
        summary_df = pd.DataFrame({
            'Change Type': list(change_types.keys()),
            'Count': list(change_types.values()),
            'Percentage': [f"{(count / max(excel_diff_count, 1) * 100):.2f}%"
                           for count in change_types.values()]
        })

        print("\nChange Type Summary:")
        print(summary_df)

        # Create a modified encoding analysis dataframe
        encoding_analysis = pd.DataFrame({
            'Analysis Type': [
                'Total Rows',
                'Differences detected by Excel EXACT()',
                'Differences detected by Python comparison',
                'Subtle encoding differences (detected by Excel but not Python)',
                'Rows where 住所カナ2 is blank',
                'Rows where 住所カナ2 contains kanji',
                'Rows where numbers changed from full-width to half-width',
                'Rows where symbols changed to kana characters',
                'Rows where symbols changed width (half to full or vice versa)',
                'Rows where 住所カナ2 contains チョウメ',
                'Rows where 住所カナ2 contains チョウ (excluding チョウメ)'
            ],
            'Count': [
                len(df),
                excel_diff_count,
                python_diff_count,
                excel_diff_count - python_diff_count,
                kana2_blank_count,
                kanji_count,
                numeric_fw_to_hw_count,
                symbol_to_kana_count,
                symbol_width_change_count,
                choume_count,
                chou_count
            ],
            'Percentage': [
                '100.00%',
                f"{(excel_diff_count / len(df) * 100):.2f}%",
                f"{(python_diff_count / len(df) * 100):.2f}%",
                f"{((excel_diff_count - python_diff_count) / len(df) * 100):.2f}%",
                f"{(kana2_blank_count / len(df) * 100):.2f}%",
                f"{(kanji_count / len(df) * 100):.2f}%",
                f"{(numeric_fw_to_hw_count / len(df) * 100):.2f}%",
                f"{(symbol_to_kana_count / len(df) * 100):.2f}%",
                f"{(symbol_width_change_count / len(df) * 100):.2f}%",
                f"{(choume_count / len(df) * 100):.2f}%",
                f"{(chou_count / len(df) * 100):.2f}%"
            ]
        })

        # Create specialized analysis sheet for new conversions
        special_analysis_df = pd.DataFrame({
            'Analysis Type': [
                'Rows where 住所カナ2 is blank',
                'Rows where 住所カナ2 contains kanji',
                'Rows where numbers changed from full-width to half-width',
                'Rows where symbols changed to kana characters',
                'Rows where symbols changed width (half to full or vice versa)'
            ],
            'Count': [
                kana2_blank_count,
                kanji_count,
                numeric_fw_to_hw_count,
                symbol_to_kana_count,
                symbol_width_change_count
            ],
            'Percentage of Total': [
                f"{(kana2_blank_count / len(df) * 100):.2f}%",
                f"{(kanji_count / len(df) * 100):.2f}%",
                f"{(numeric_fw_to_hw_count / len(df) * 100):.2f}%",
                f"{(symbol_to_kana_count / len(df) * 100):.2f}%",
                f"{(symbol_width_change_count / len(df) * 100):.2f}%"
            ],
            'Percentage of Changes': [
                f"{(kana2_blank_count / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(kanji_count / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(numeric_fw_to_hw_count / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(symbol_to_kana_count / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(symbol_width_change_count / max(excel_diff_count, 1) * 100):.2f}%"
            ]
        })

        # Save all results to separate sheets in the output file
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detailed Comparison', index=False)
            summary_df.to_excel(writer, sheet_name='Change Summary', index=False)
            encoding_analysis.to_excel(writer, sheet_name='Encoding Analysis', index=False)
            special_analysis_df.to_excel(writer, sheet_name='Special Analysis', index=False)

            # Create sheets for examples of each special case
            if kana2_blank_count > 0:
                blank_examples = df[df['kana2_is_blank']].head(100).copy()
                blank_examples.to_excel(writer, sheet_name='Blank Kana2 Examples', index=False)

            if kanji_count > 0:
                kanji_examples = df[df['contains_kanji']].head(100).copy()
                kanji_examples.to_excel(writer, sheet_name='Kanji Examples', index=False)

            if numeric_fw_to_hw_count > 0:
                numeric_examples = df[df['numeric_fullwidth_to_halfwidth']].head(100).copy()
                numeric_examples.to_excel(writer, sheet_name='FW to HW Examples', index=False)

            if symbol_to_kana_count > 0:
                symbol_examples = df[df['symbol_to_kana']].head(100).copy()
                symbol_examples.to_excel(writer, sheet_name='Symbol to Kana Examples', index=False)

            if symbol_width_change_count > 0:
                width_examples = df[df['symbol_width_change']].head(100).copy()
                width_examples.to_excel(writer, sheet_name='Symbol Width Change Examples', index=False)

        print(f"\nSaving results with summary to {output_path}...")
        print("Done!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

        # Try to help diagnose common errors
        if "No such file or directory" in str(e):
            print(f"Could not find the file at {file_path}")
            print("Make sure your Excel file is at that exact location.")
        elif "Unsupported format" in str(e):
            print("Your file might not be a valid Excel file.")
            print("Make sure it's saved as .xlsx format.")


if __name__ == "__main__":
    main()