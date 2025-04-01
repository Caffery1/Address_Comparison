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


def has_numeric_halfwidth_to_fullwidth(original, modified):
    """
    Check if numbers changed from half-width to full-width
    """
    if pd.isna(original) or pd.isna(modified):
        return False

    original_str = str(original)
    modified_str = str(modified)

    # Create regex pattern to find half-width digits in original
    hw_digits_pattern = re.compile(r'[0-9]')

    # Dictionary mapping half-width to full-width digits
    hw_to_fw_digits = {
        '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
        '5': '５', '6': '６', '7': '７', '8': '８', '9': '９'
    }

    # Find all half-width digits in original
    hw_digits_matches = hw_digits_pattern.finditer(original_str)

    for match in hw_digits_matches:
        idx = match.start()
        hw_digit = original_str[idx]

        # Check if this index exists in modified and contains the full-width equivalent
        if idx < len(modified_str) and modified_str[idx] == hw_to_fw_digits.get(hw_digit):
            return True

    # If we find half-width in original and full-width in modified text at exact positions
    d = difflib.SequenceMatcher(None, original_str, modified_str)
    for tag, i1, i2, j1, j2 in d.get_opcodes():
        if tag == 'replace':
            orig_chars = original_str[i1:i2]
            mod_chars = modified_str[j1:j2]

            # Check if this replacement includes half-width to full-width digit conversion
            for i, c1 in enumerate(orig_chars):
                if i < len(mod_chars) and c1 in hw_to_fw_digits and mod_chars[i] == hw_to_fw_digits[c1]:
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
        # Maps for half-width to full-width conversions
        hw_to_fw = {
            '0': '０', '1': '１', '2': '２', '3': '３', '4': '４',
            '5': '５', '6': '６', '7': '７', '8': '８', '9': '９',
            '-': 'ー', '.': '．', '(': '（', ')': '）', ' ': '　',
            '/': '／', ',': '，', ':': '：'
        }

        # If lengths are different, can't be just width difference
        if len(str1) != len(str2):
            return False

        for c1, c2 in zip(str1, str2):
            # If characters are the same, continue
            if c1 == c2:
                continue

            # Check if it's a half-width to full-width conversion
            if hw_to_fw.get(c1) == c2 or hw_to_fw.get(c2) == c1:
                continue

            # If we get here, found a difference that isn't just width
            return False

        # If we get through the whole string, it's only width differences
        return True

    # Check for specific encoding differences
    if is_only_width_different(original, modified):
        # Get the specific characters that differ
        diff_pairs = []
        for i, (c1, c2) in enumerate(zip(original, modified)):
            if c1 != c2:
                diff_pairs.append(f"Position {i}: '{c1}' → '{c2}'")

        return f"Only half-width/full-width differences: {'; '.join(diff_pairs[:3])}" + \
            (f"; and {len(diff_pairs) - 3} more" if len(diff_pairs) > 3 else "")

    # Check for unicode normalization differences
    if unicodedata.normalize('NFKC', original) == unicodedata.normalize('NFKC', modified):
        return "Unicode normalization difference"

    # Check for hidden/invisible characters
    def contains_invisible(text):
        invisible_chars = [
            '\u200B', '\u200C', '\u200D', '\u2060',  # Zero-width spaces/joiners
            '\u00A0', '\u2000', '\u2001', '\u2002', '\u2003',  # Various spaces
            '\u2028', '\u2029',  # Line/paragraph separators
            '\uFEFF'  # BOM
        ]
        return any(c in text for c in invisible_chars)

    if contains_invisible(original) or contains_invisible(modified):
        # Try to find and highlight invisible characters
        original_invisible = [f"Pos {i}: {ord(c)}" for i, c in enumerate(original)
                              if c in ['\u200B', '\u200C', '\u200D', '\u2060', '\u00A0',
                                       '\u2000', '\u2001', '\u2002', '\u2003', '\u2028',
                                       '\u2029', '\uFEFF']]
        modified_invisible = [f"Pos {i}: {ord(c)}" for i, c in enumerate(modified)
                              if c in ['\u200B', '\u200C', '\u200D', '\u2060', '\u00A0',
                                       '\u2000', '\u2001', '\u2002', '\u2003', '\u2028',
                                       '\u2029', '\uFEFF']]

        return f"Contains invisible characters - Original: {original_invisible if original_invisible else 'none'}, Modified: {modified_invisible if modified_invisible else 'none'}"

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

        # Add a column to track exact() function behavior
        # First apply Excel's EXACT() logic by checking binary equality
        import numpy as np

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

        # Add new columns for kanji detection and half-width to full-width numeric conversion
        df['contains_kanji'] = df[kana2_column].apply(contains_kanji)
        df['numeric_halfwidth_to_fullwidth'] = df.apply(
            lambda row: has_numeric_halfwidth_to_fullwidth(row[kana1_column], row[kana2_column]),
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
        numeric_hw_to_fw_count = df['numeric_halfwidth_to_fullwidth'].sum()
        choume_count = df['contains_choume'].sum()
        chou_count = df['contains_chou'].sum()

        print(f"\nExcel EXACT() would find {excel_diff_count} differences")
        print(f"Python string comparison finds {python_diff_count} differences")
        print(f"Difference: {excel_diff_count - python_diff_count} rows have subtle encoding differences")
        print(f"Found {kanji_count} rows where 住所カナ2 contains kanji")
        print(f"Found {numeric_hw_to_fw_count} rows where numbers changed from half-width to full-width")
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

        # Analyze types of changes
        print("\nSummarizing types of changes...")

        # Initialize counters for different types of changes
        change_types = {
            'added': 0,  # New text added
            'removed': 0,  # Text removed
            'replaced': 0,  # Text replaced with something else
            'numeric_added': 0,  # Numbers added
            'symbols_added': 0,  # Symbols added (like -, —, etc.)
            'format_change': 0,  # Same content but different format (e.g., "6-2" vs "６ー２")
            'whitespace_only': 0,  # Only whitespace differences
            'half_full_width_only': 0,  # Only half/full-width character differences
            'unicode_normalization': 0,  # Unicode normalization differences
            'invisible_characters': 0,  # Invisible/hidden character differences
            'exact_match_in_python': 0,  # No difference detected in Python but different in Excel
            'contains_kanji': kanji_count,  # Number of rows where 住所カナ2 contains kanji
            'numeric_halfwidth_to_fullwidth': numeric_hw_to_fw_count,
            # Number of rows with half to full-width conversion
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

                # Count specific types of encoding differences
                if 'half-width/full-width differences' in diff_text:
                    change_types['half_full_width_only'] += 1

                elif 'Whitespace difference' in diff_text:
                    change_types['whitespace_only'] += 1

                elif 'Unicode normalization difference' in diff_text:
                    change_types['unicode_normalization'] += 1

                elif 'invisible characters' in diff_text:
                    change_types['invisible_characters'] += 1

                # Count substantive differences
                elif 'Added:' in diff_text:
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
                    # Check if it's just a format change (half-width to full-width or vice versa)
                    if re.search(r"Changed: '[-0-9]+' → '[－０-９]+'", diff_text) or \
                            re.search(r"Changed: '[－０-９]+' → '[-0-9]+'", diff_text):
                        change_types['format_change'] += 1

            else:
                # This should never happen if our comparison is correct,
                # but count cases where Python thinks they're the same but they differ in Excel
                change_types['exact_match_in_python'] += 1

        # For the rows that Excel finds different but Python doesn't, analyze further
        if excel_diff_count > python_diff_count:
            print("\nAnalyzing subtle encoding differences that Excel EXACT() would detect but Python missed...")

            # Sample a few rows with subtle differences
            subtle_diff_rows = df[df['excel_python_diff']].head(5)

            for idx, row in subtle_diff_rows.iterrows():
                orig = row[kana1_column]
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ1: {orig}")
                print(f"カナ2: {mod}")
                print(f"Length カナ1: {len(str(orig))}, Length カナ2: {len(str(mod))}")

                # Compare character codes
                orig_codes = [f"{c}({ord(c)})" for c in str(orig)]
                mod_codes = [f"{c}({ord(c)})" for c in str(mod)]

                # Only show differences in character codes
                code_diffs = []
                for i, (o, m) in enumerate(zip(orig_codes, mod_codes)):
                    if o != m:
                        code_diffs.append(f"Position {i}: {o} vs {m}")

                if code_diffs:
                    print("Character code differences:")
                    for diff in code_diffs[:5]:  # Limit to first 5 differences
                        print(f"  {diff}")
                    if len(code_diffs) > 5:
                        print(f"  ...and {len(code_diffs) - 5} more differences")
                else:
                    print("No character code differences found despite binary inequality")

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

        # Detailed analysis of numeric half-width to full-width conversions
        if numeric_hw_to_fw_count > 0:
            print(f"\nAnalyzing {numeric_hw_to_fw_count} rows where numbers changed from half-width to full-width...")

            # Sample a few rows with numeric conversions
            numeric_rows = df[df['numeric_halfwidth_to_fullwidth']].head(5)

            for idx, row in numeric_rows.iterrows():
                orig = row[kana1_column]
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ1: {orig}")
                print(f"カナ2: {mod}")

                # Find half-width digits in original
                hw_digits = re.findall(r'[0-9]', str(orig))
                # Find full-width digits in modified
                fw_digits = re.findall(r'[０-９]', str(mod))

                print(f"Half-width digits in カナ1: {hw_digits}")
                print(f"Full-width digits in カナ2: {fw_digits}")

        # Detailed analysis of rows containing 'チョウメ'
        if choume_count > 0:
            print(f"\nAnalyzing {choume_count} rows where 住所カナ2 contains 'チョウメ'...")

            # Sample a few rows with 'チョウメ'
            choume_rows = df[df['contains_choume']].head(5)

            for idx, row in choume_rows.iterrows():
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ2: {mod}")

                # Find the position of 'チョウメ'
                match = re.search(r'チョウメ', str(mod))
                if match:
                    start_pos = match.start()
                    end_pos = match.end()
                    print(f"'チョウメ' found at positions {start_pos}-{end_pos}")

                    # Try to get some context around 'チョウメ'
                    start_context = max(0, start_pos - 10)
                    end_context = min(len(str(mod)), end_pos + 10)
                    print(f"Context: ...{str(mod)[start_context:end_context]}...")

        # Detailed analysis of rows containing 'チョウ' (but not 'チョウメ')
        if chou_count > 0:
            print(f"\nAnalyzing {chou_count} rows where 住所カナ2 contains 'チョウ' (excluding 'チョウメ')...")

            # Sample a few rows with 'チョウ'
            chou_rows = df[df['contains_chou']].head(5)

            for idx, row in chou_rows.iterrows():
                mod = row[kana2_column]

                print(f"\nRow {idx}:")
                print(f"カナ2: {mod}")

                # Find the position of 'チョウ'
                match = re.search(r'チョウ', str(mod))
                if match:
                    start_pos = match.start()
                    end_pos = match.end()
                    print(f"'チョウ' found at positions {start_pos}-{end_pos}")

                    # Try to get some context around 'チョウ'
                    start_context = max(0, start_pos - 10)
                    end_context = min(len(str(mod)), end_pos + 10)
                    print(f"Context: ...{str(mod)[start_context:end_context]}...")

        # Create a summary dataframe
        summary_df = pd.DataFrame({
            'Change Type': list(change_types.keys()),
            'Count': list(change_types.values()),
            'Percentage': [f"{(count / max(excel_diff_count, 1) * 100):.2f}%"
                           for count in change_types.values()]
        })

        print("\nChange Type Summary:")
        print(summary_df)

        # Add supplementary analysis to identify invisible characters
        print("\nChecking for invisible characters in all data...")

        def detect_invisible_chars(text):
            if pd.isna(text):
                return []

            text = str(text)
            invisible_chars = {
                '\u200B': 'Zero Width Space',
                '\u200C': 'Zero Width Non-Joiner',
                '\u200D': 'Zero Width Joiner',
                '\u2060': 'Word Joiner',
                '\u00A0': 'Non-Breaking Space',
                '\u2028': 'Line Separator',
                '\u2029': 'Paragraph Separator',
                '\uFEFF': 'Byte Order Mark',
                '\u3000': 'Ideographic Space',
            }

            found = []
            for i, char in enumerate(text):
                if char in invisible_chars:
                    found.append(f"Position {i}: {invisible_chars[char]} (U+{ord(char):04X})")
            return found

        # Check both columns for invisible characters
        invisible_in_kana1 = df[kana1_column].apply(detect_invisible_chars)
        invisible_in_kana2 = df[kana2_column].apply(detect_invisible_chars)

        # Count rows with invisible characters
        rows_with_invisible = (invisible_in_kana1.apply(len) > 0) | (invisible_in_kana2.apply(len) > 0)
        invisible_count = rows_with_invisible.sum()

        if invisible_count > 0:
            print(f"Found {invisible_count} rows with invisible characters")

            # Add to summary
            if 'invisible_characters' in change_types:
                change_types['invisible_characters'] = invisible_count

            # Create a sheet with examples of rows with invisible characters
            invisible_examples = df[rows_with_invisible].head(100).copy()
            invisible_examples['invisible_in_kana1'] = invisible_in_kana1[rows_with_invisible].head(100)
            invisible_examples['invisible_in_kana2'] = invisible_in_kana2[rows_with_invisible].head(100)

        # Create a special analysis for encoding differences
        encoding_analysis = pd.DataFrame({
            'Analysis Type': [
                'Total Rows',
                'Differences detected by Excel EXACT()',
                'Differences detected by Python comparison',
                'Subtle encoding differences (detected by Excel but not Python)',
                'Half-width/Full-width character differences',
                'Whitespace differences',
                'Unicode normalization differences',
                'Invisible character differences',
                'Character code differences',
                'Rows where 住所カナ2 contains kanji',
                'Rows where numbers changed from half-width to full-width',
                'Rows where 住所カナ2 contains チョウメ',
                'Rows where 住所カナ2 contains チョウ (excluding チョウメ)'
            ],
            'Count': [
                len(df),
                excel_diff_count,
                python_diff_count,
                excel_diff_count - python_diff_count,
                change_types['half_full_width_only'],
                change_types['whitespace_only'],
                change_types['unicode_normalization'],
                change_types['invisible_characters'],
                sum(change_types[k] for k in ['half_full_width_only', 'whitespace_only',
                                              'unicode_normalization', 'invisible_characters']),
                kanji_count,
                numeric_hw_to_fw_count,
                choume_count,
                chou_count
            ],
            'Percentage': [
                '100.00%',
                f"{(excel_diff_count / len(df) * 100):.2f}%",
                f"{(python_diff_count / len(df) * 100):.2f}%",
                f"{((excel_diff_count - python_diff_count) / len(df) * 100):.2f}%",
                f"{(change_types['half_full_width_only'] / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(change_types['whitespace_only'] / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(change_types['unicode_normalization'] / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(change_types['invisible_characters'] / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(sum(change_types[k] for k in ['half_full_width_only', 'whitespace_only',
                                                  'unicode_normalization', 'invisible_characters']) / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(kanji_count / len(df) * 100):.2f}%",
                f"{(numeric_hw_to_fw_count / len(df) * 100):.2f}%"
            ]
        })

        # Create specialized analysis sheet for kanji and numeric conversions
        special_analysis_df = pd.DataFrame({
            'Analysis Type': [
                'Rows where 住所カナ2 contains kanji',
                'Rows where numbers changed from half-width to full-width'
            ],
            'Count': [
                kanji_count,
                numeric_hw_to_fw_count
            ],
            'Percentage of Total': [
                f"{(kanji_count / len(df) * 100):.2f}%",
                f"{(numeric_hw_to_fw_count / len(df) * 100):.2f}%"
            ],
            'Percentage of Changes': [
                f"{(kanji_count / max(excel_diff_count, 1) * 100):.2f}%",
                f"{(numeric_hw_to_fw_count / max(excel_diff_count, 1) * 100):.2f}%"
            ]
        })

        # Save all results to separate sheets in the output file
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Detailed Comparison', index=False)
            summary_df.to_excel(writer, sheet_name='Change Summary', index=False)
            encoding_analysis.to_excel(writer, sheet_name='Encoding Analysis', index=False)
            special_analysis_df.to_excel(writer, sheet_name='Kanji and Numeric Analysis', index=False)

            # If we found invisible characters, add that sheet too
            if invisible_count > 0:
                invisible_examples.to_excel(writer, sheet_name='Invisible Characters', index=False)

            # Create sheets for kanji and numeric examples
            if kanji_count > 0:
                kanji_examples = df[df['contains_kanji']].head(100).copy()
                kanji_examples.to_excel(writer, sheet_name='Kanji Examples', index=False)

            if numeric_hw_to_fw_count > 0:
                numeric_examples = df[df['numeric_halfwidth_to_fullwidth']].head(100).copy()
                numeric_examples.to_excel(writer, sheet_name='Numeric HW to FW Examples', index=False)

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