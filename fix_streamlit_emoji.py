#!/usr/bin/env python3
"""
Fix emoji characters in Streamlit apps for Windows compatibility
"""

import re
from pathlib import Path

# Emoji to ASCII replacement map
EMOJI_MAP = {
    # Medical/Health
    '🫀': '[HEART]',
    '❤️': '[HEART]',
    '💓': '[HEART]',
    '🏥': '[HOSPITAL]',
    '🩺': '[STETHOSCOPE]',
    '💊': '[PILL]',
    '🔬': '[MICROSCOPE]',

    # UI Elements
    '📊': '[CHART]',
    '📈': '[GRAPH]',
    '📉': '[DOWNTREND]',
    '🎯': '[TARGET]',
    '🧠': '[BRAIN]',
    '⚡': '[LIGHTNING]',
    '✨': '[SPARKLES]',
    '🔍': '[SEARCH]',
    '🔎': '[SEARCH]',

    # Status indicators
    '✅': '[OK]',
    '❌': '[ERROR]',
    '⚠️': '[WARNING]',
    '✓': '[CHECK]',
    '✗': '[X]',
    '🚀': '[ROCKET]',
    '📁': '[FOLDER]',
    '📝': '[MEMO]',
    '💡': '[BULB]',

    # Other
    '🎨': '[PALETTE]',
    '🌟': '[STAR]',
    '📌': '[PIN]',
    '🔔': '[BELL]',
    '⭐': '[STAR]',
    '🔥': '[FIRE]',
}

def fix_emojis_in_file(filepath):
    """Remove/replace emojis in a file"""
    print(f"Processing: {filepath.name}")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Replace known emojis
    for emoji, replacement in EMOJI_MAP.items():
        if emoji in content:
            content = content.replace(emoji, replacement)
            print(f"  Replaced emoji with {replacement}")

    # Remove any remaining non-ASCII characters in strings
    # This is a more aggressive approach for any missed emojis
    def replace_emoji(match):
        text = match.group(1)
        # Replace remaining non-ASCII with empty string
        clean_text = ''.join(char if ord(char) < 128 else '' for char in text)
        return f'"{clean_text}"' if match.group(0).startswith('"') else f"'{clean_text}'"

    # Match quoted strings and remove emojis
    content = re.sub(r'["\']([^"\']*)["\']', replace_emoji, content)

    if content != original_content:
        # Backup original
        backup_path = filepath.with_suffix('.py.backup')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"  Backup saved: {backup_path.name}")

        # Write fixed version
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Fixed file saved")
        return True
    else:
        print(f"  No emojis found")
        return False

def main():
    print("=" * 70)
    print("FIXING STREAMLIT EMOJI ENCODING ISSUES")
    print("=" * 70)

    files_to_fix = [
        'dashboard.py',
        'patient_analysis_app.py',
        'enhanced_xai_app.py'
    ]

    fixed_count = 0

    for filename in files_to_fix:
        filepath = Path(filename)
        if filepath.exists():
            if fix_emojis_in_file(filepath):
                fixed_count += 1
        else:
            print(f"File not found: {filename}")

    print("\n" + "=" * 70)
    print(f"COMPLETE: Fixed {fixed_count}/{len(files_to_fix)} files")
    print("=" * 70)
    print("\nBackup files created with .py.backup extension")
    print("Test the apps, then delete backups if everything works")

if __name__ == "__main__":
    main()
