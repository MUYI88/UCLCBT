pragma solidity ^0.4.24;

import './erc20interface.sol';

contract ERC20A is ERC20Interface {


    string public  name;//DFreda
    string public constant symbol = "DFR";
    uint8 public constant decimals = 18;  // 18 is the most common number of decimal places
    
    uint public totalSupply;//1000000
     
    mapping(address => uint256) internal _balances;

    mapping(address => mapping(address => uint256)) allowed;

    constructor(string memory _name) public {
       name = _name;  // "DFreda";
       totalSupply = 1000000*10**uint256(18);
       _balances[msg.sender] = totalSupply;
    }

    function balanceOf(address tokenOwner) public view returns (uint balance) {
        return _balances[tokenOwner];
    }

  function transfer(address _to, uint256 _value)  public returns (bool success) {
      require(_to != address(0));
      require(_balances[msg.sender] >= _value);
      require(_balances[ _to] + _value >= _balances[ _to]);   

      _balances[msg.sender] -= _value;
      _balances[_to] += _value;

      emit Transfer(msg.sender, _to, _value);

      return true;
    }

  function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
      require(_to != address(0));
      require(allowed[_from][msg.sender] >= _value);
      require(_balances[_from] >= _value);
      require(_balances[ _to] + _value >= _balances[ _to]);

      _balances[_from] -= _value;
      _balances[_to] += _value;

      allowed[_from][msg.sender] -= _value;

      emit Transfer(msg.sender, _to, _value);
      return true;
    }

  function approve(address _spender, uint256 _value) public returns (bool success) {
      allowed[msg.sender][_spender] = _value;

      emit Approval(msg.sender, _spender, _value);
      return true;
    }

  function allowance(address _owner, address _spender) public view returns (uint256 remaining) {
      return allowed[_owner][_spender];
    }

}
