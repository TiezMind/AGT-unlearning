import hydra
from src import it_unlearn
import debugpy

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

@hydra.main(version_base=None, config_path="config", config_name="forget_lora")
def main(cfg):
    it_unlearn(cfg)

if __name__ == "__main__":
    main()
