import sys

from regression_core import (
    check_cloud_stack,
    get_default_model,
    run_regression,
    verify_output,
)


def main():
    print("=== YouTube-to-Docs Regression: SharePoint Storage ===")

    # 2. Check Cloud Stack
    available_models = check_cloud_stack()
    if not available_models:
        print("No models available. Please check your environment variables.")
        sys.exit(1)

    # 3. Select Model
    selected_model = get_default_model(available_models)
    print(f"\nUsing model: {selected_model}")

    # 4. Run Regression with SharePoint Target
    run_regression(
        "foundry-gpt-5-mini",  # model
        None,  # transcript_model
        None,  # infographic_model
        None,  # tts_model
        language="en",
        output_target="sharepoint",
        all_gemini_arg=None,
    )

    # 5. Verify Output from SharePoint
    verify_output(
        "foundry-gpt-5-mini",
        None,
        None,
        None,
        language="en",
        output_target="sharepoint",
        all_gemini_arg=None,
    )

    print("\n=== SUCCESS: SharePoint Storage Regression Passed ===")


if __name__ == "__main__":
    main()
