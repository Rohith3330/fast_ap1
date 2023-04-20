from pydantic import BaseModel


class GAN(BaseModel):
   type : int
   image : str
class sample(BaseModel):
   type:int
   image:list