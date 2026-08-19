import os
from pathlib import Path
from PIL import Image


class BatchImageConverter:
    """Class to manage batch conversion of images from a source directory

    to a destination directory while preserving folder hierarchy.
    """

    def __init__(
        self,
        source_dir: str,
        output_dir: str,
        target_extensions: tuple = (
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
            ".png",
        ),
    ):
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_extensions = tuple(
            ext.lower() for ext in target_extensions
        )

    def convert_image_to_png(
        self, src_file: Path, dest_file: Path
    ) -> bool:
        """Converts a single image file to PNG format."""
        try:
            # Create subdirectories in destination if they don't exist
            dest_file.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(src_file) as img:
                # Convert palette/grayscale/CMYK modes to RGB before saving as PNG
                if img.mode in ("P", "CMYK", "1"):
                    img = img.convert("RGB")
                elif img.mode == "RGBA":
                    # Keep transparency if present
                    pass

                # Change extension to .png
                final_output_path = dest_file.with_suffix(".png")
                img.save(final_output_path, "PNG")
                print(f"[SUCCESS] Converted: {src_file.name} -> {final_output_path.name}")
                return True

        except Exception as e:
            print(f"[ERROR] Failed to convert {src_file}: {e}")
            return False

    def process_all_directories(self) -> None:
        """Traverses source_dir recursively and processes matching files."""
        if not self.source_dir.exists():
            print(f"Source directory '{self.source_dir}' does not exist.")
            return

        converted_count = 0
        skipped_count = 0

        # os.walk traverses all subfolders automatically
        for root, _, files in os.walk(self.source_dir):
            current_path = Path(root)

            for file_name in files:
                file_path = current_path / file_name

                if file_path.suffix.lower() in self.target_extensions:
                    # Maintain relative path structure inside target directory
                    relative_path = file_path.relative_to(self.source_dir)
                    destination_path = self.output_dir / relative_path

                    if self.convert_image_to_png(file_path, destination_path):
                        converted_count += 1
                    else:
                        skipped_count += 1

        print("\n--- Conversion Finished ---")
        print(f"Total converted: {converted_count}")
        print(f"Total failed/skipped: {skipped_count}")


# --- Object Instantiation & Usage ---
if __name__ == "__main__":
    SOURCE_FOLDER = "..\\Output\\contrast_enhancement"
    OUTPUT_FOLDER = "..\\Output\\data_png"

    # Instantiate the converter object
    converter = BatchImageConverter(
        source_dir=SOURCE_FOLDER, output_dir=OUTPUT_FOLDER
    )

    # Execute the process
    converter.process_all_directories()