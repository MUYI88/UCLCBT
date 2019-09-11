pragma solidity ^0.4.24;


import './owned.sol';
import './erc20a.sol';

contract AdvanceToken is ERC20A, owned {
    
    uint256 public sellPrice;
    uint256 public buyPrice;
    
    mapping (address => bool) public frozenAccount;

    event AddSupply(uint amount);
    event FrozenFunds(address target, bool frozen);
    event Burn(address target, uint amount);

    constructor (string _name) ERC20A(_name) public {

    }

    function mine(address target, uint amount) public onlyOwner {
        totalSupply += amount;
        _balances[target] += amount;

        emit AddSupply(amount);
        emit Transfer(address(0), target, amount);
    }

    function freezeAccount(address target, bool freeze) public onlyOwner {
        frozenAccount[target] = freeze;
        emit FrozenFunds(target, freeze);
    }


    function transfer(address _to, uint256 _value)  public returns (bool success) {
      require(_to != address(0));
      require(!frozenAccount[msg.sender]);
      require(_balances[msg.sender] >= _value);
      require(_balances[ _to] + _value >= _balances[ _to]);  
      _balances[msg.sender] -= _value;
      _balances[_to] += _value;

      emit Transfer(msg.sender, _to, _value);

      return true;
    }
    
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
      require(_to != address(0));
      require(!frozenAccount[_from]);
      require(allowed[_from][msg.sender] >= _value);
      require(_balances[_from] >= _value);
      require(_balances[ _to] + _value >= _balances[ _to]);

      _balances[_from] -= _value;
      _balances[_to] += _value;

      allowed[_from][msg.sender] -= _value;

      emit Transfer(msg.sender, _to, _value);
      return true;
    }

    function burn(uint256 _value) public onlyOwner returns (bool success) {
        require(_balances[msg.sender] >= _value);

        totalSupply -= _value;
        _balances[msg.sender] -= _value;

        emit Burn(msg.sender, _value);
        return true;
    }

    function burnFrom(address _from, uint256 _value)  public onlyOwner returns (bool success) {
        require(_balances[_from] >= _value);
        require(allowed[_from][msg.sender] >= _value);

        totalSupply -= _value;
        _balances[msg.sender] -= _value;
        allowed[_from][msg.sender] -= _value;

        emit Burn(msg.sender, _value);
        return true;
    }
    
    
    function setPrices(uint256 newSellPrice, uint256 newBuyPrice) onlyOwner public {
        sellPrice = newSellPrice;
        buyPrice = newBuyPrice;
    }

    
    function buy() payable public returns (uint amount) {
        amount = msg.value / buyPrice;                            
        require(_balances[this] >= amount);   
        _balances[msg.sender] += amount;
        _balances[this] -=amount;
        emit Transfer(this, msg.sender, amount);              
        return amount;
    }

    
    function sell(uint256 amount) public returns (uint256 revenue) {
       require(_balances[msg.sender] >= amount);
       _balances[this] +=amount;
       _balances[msg.sender] -= amount;
       revenue = amount * sellPrice;
       msg.sender.transfer(revenue);
       emit Transfer (msg.sender, this, amount);
       return revenue;
       
    }
}
