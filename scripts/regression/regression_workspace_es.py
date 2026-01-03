import sys

from regression_core import (
    check_cloud_stack,
    get_default_model,
    resolve_drive_folder,
    run_regression,
    verify_output,
)


def main():
    print(
        "=== YouTube-to-Docs Regression: "
        "Google Workspace Storage (Spanish, Gemini Pro) ==="
    )

    # 1. Resolve Drive Folder (Delete if exists, then Recreate
    folder_id = resolve_drive_folder(
        "youtube-to-docs-test-drive", delete_if_exists=True
    )
    print(f"Using Drive Folder ID: {folder_id}")

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing model: {selected_model}")

    # 4. Run Regression with Workspace Target
    run_regression(
        None,  # model will be set by all_gemini_arg
        None,  # transcript_model
        None,  # infographic_model
        None,  # tts_model
        language="es",
        output_target=folder_id,
        all_gemini_arg="gemini-pro",
    )

    # 5. Verify Output from Workspace
    verify_output(
        None,
        None,
        None,
        None,
        language="es",
        output_target=folder_id,
        all_gemini_arg="gemini-pro",
    )

    print("\n=== SUCCESS: Workspace Storage Regression (Spanish) Passed ===")


if __name__ == "__main__":
    main()
