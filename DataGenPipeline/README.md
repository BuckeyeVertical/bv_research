# DataGenPipeline

This tool is designed to augment image datasets for object detection model training. It processes a directory of input images and synthetically adds "tents" to them, helping to create more robust training data for identifying these objects in aerial or ground imagery.

## Getting Started

Follow the steps below to set up the environment and run the data generation pipeline.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BuckeyeVertical/bv_research.git](https://github.com/BuckeyeVertical/bv_research.git)
    ```

2.  **Navigate to the pipeline directory:**
    ```bash
    cd bv_research/DataGenPipeline
    ```

3.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```

4.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```
    *(Note: On Windows, use `.venv\Scripts\activate` instead)*

5.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the pipeline, execute `main.py` and pass the path to the directory containing your images as an argument:

```bash
python main.py <path_to_folder_with_images>
